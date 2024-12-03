import subprocess
import polars as pl
import tempfile
import os
from typing import List, Optional, Dict
from pathlib import Path
from .lookups import ROUND_DAYS_LOOKUP, SURFACE_LOOKUP

class TennisLoader:
    """A class to load and process tennis MDB database files with cached tables."""
    
    def __init__(self, mdb_file: str, password: str):
        """
        Initialize the MDB loader and load common tables.
        
        Args:
            mdb_file: Path to the .mdb file
            password: Database password
        """
        self._round_days_lookup = ROUND_DAYS_LOOKUP
        self._surface_lookup = SURFACE_LOOKUP

        self.mdb_file = Path(mdb_file)
        self.password = password
        self._tables: Dict[str, pl.DataFrame] = {}
        self.games = None
        self.players: Dict[int, str] = {}
        self.mindate = None
        
        if not self.mdb_file.exists():
            raise FileNotFoundError(f"MDB file not found: {mdb_file}")
        
        # Load essential tables on initialization and do initial filter
        self.load_tables(['players_atp', 'games_atp', 'tours_atp', 'players_wta', 'games_wta', 'tours_wta', 'courts', 'rounds'])
        # Get a dict of players to codes and filter games with defaults only
        self.load_players()
            
    def _get_env(self) -> dict:
        """Create environment dictionary with MDB password."""
        env = os.environ.copy()
        env['MDB_JET_PASSWORD'] = self.password
        return env
    
    def get_tables(self) -> List[str]:
        """Get list of tables from the password-protected MDB file."""
        try:
            result = subprocess.run(
                ['mdb-tables', str(self.mdb_file)], 
                capture_output=True, 
                text=True, 
                check=True,
                env=self._get_env()
            )
            tables = result.stdout.strip().split(' ')
            return [table for table in tables if table]
            
        except subprocess.CalledProcessError as e:
            print(f"Error getting tables: {str(e)}")
            return []
        except FileNotFoundError:
            print("Error: mdbtools is not installed. Please install it first.")
            print("Ubuntu/Debian: sudo apt-get install mdbtools")
            print("MacOS: brew install mdbtools")
            return []
    
    def load_tables(self, table_names: List[str]) -> None:
        """
        Load specified tables into memory.
        
        Args:
            table_names: List of table names to load
        """
        for table_name in table_names:
            df = self.read_table(table_name)
            if df is not None:
                self._tables[table_name] = df
            else:
                print(f"Warning: Failed to load table {table_name}")

    def load_players(self):
        """
        Get player information by ID.
        
        Args:
            player_id: Player ID to look up
            
        Returns:
            Dictionary containing player information or None if not found
        """
        players_df = self.get_table('players_atp')
        if players_df is None:
            raise ValueError("Players table not loaded")
            
        player_dict = dict(zip(
            players_df.get_column('ID_P').to_list(),
            players_df.get_column('NAME_P').to_list()
            ))
            
        self.players = player_dict

    def clean_games(self, singles_doubles = 1, guess_dates = True, add_surfaces = True):
        """
        Clean down the games from the tables down to the ones
        we want to fit. Will include dates, tournaments
        etc at some point. Right now, just does singles/doubles.
        """
        df = self._tables['games_atp']

        df = df.rename({
        'ID1_G': 'P1',
        'ID2_G': 'P2',
        'ID_T_G': 'Tour',
        'ID_R_G': 'Round',
        'RESULT_G': 'Result',
        'DATE_G': 'Date_STR'
        })

        if add_surfaces:
            #Lookup from tours table
            df = df.join(
            self._tables['tours_atp'].select(['ID_T', 'ID_C_T']),  # pre-select just the columns we need
            left_on='Tour',
            right_on='ID_T',
            how='left').select(pl.all().exclude(['ID_T']))
            df = df.rename({'ID_C_T':'Surface'})
            #Aggregate into surface IDs
            df = df.with_columns(pl.col('Surface').map_elements(lambda x: self._surface_lookup.get(x, None), return_dtype=pl.Int8))

        if guess_dates:
            #Lookup from tours table
            df = df.join(
            self._tables['tours_atp'].select(['ID_T', 'DATE_T']),  # pre-select just the columns we need
            left_on='Tour', right_on='ID_T', how='left').select(pl.all().exclude(['ID_T']))
            df = df.rename({'DATE_T':'Tour_date_STR'})

        df, self.mindate = self._dates_to_daynums(df)

        filtered_df = (
            df.filter(pl.col('P1').map_elements(lambda x: self.players.get(x, None), return_dtype=pl.Utf8).fill_null('').str.contains('/').not_()) if singles_doubles == 1
            else df.filter(pl.col('P2').map_elements(lambda x: self.players.get(x, None), return_dtype=pl.Utf8).fill_null('').str.contains('/')) if singles_doubles == 2
            else df)
        
        filtered_df = filtered_df.with_columns(pl.col('Final_date').dt.date().alias('Date')).select(['P1', 'P2', 'Tour', 'Surface', 'Day', 'Date'])        
        
        self.games = filtered_df
    
    def get_table(self, table_name: str) -> Optional[pl.DataFrame]:
        """
        Get a table from memory, loading it if necessary.
        
        Args:
            table_name: Name of the table to retrieve
            
        Returns:
            Polars DataFrame containing the table data, or None if table doesn't exist
        """
        if table_name not in self._tables:
            df = self.read_table(table_name)
            if df is not None:
                self._tables[table_name] = df
        return self._tables.get(table_name)
    
    def read_table(self, table_name: str) -> Optional[pl.DataFrame]:
        """Read a specific table from the MDB file into a Polars DataFrame."""
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as temp_file:
                temp_path = temp_file.name
            
            subprocess.run(
                ['mdb-export', str(self.mdb_file), table_name], 
                stdout=open(temp_path, 'w'),
                check=True,
                env=self._get_env()
            )
            
            df = pl.read_csv(temp_path)
            return df
            
        except subprocess.CalledProcessError as e:
            print(f"Error reading table {table_name}: {str(e)}")
            return None
        except Exception as e:
            print(f"Error processing table {table_name}: {str(e)}")
            return None
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def _dates_to_daynums(self, df: pl.DataFrame) -> pl.DataFrame:
        """Convert string dates to datetime and add daynum column."""

        df = df.with_columns(
            pl.col('Tour_date_STR')
            .str.strptime(pl.Datetime, format="%m/%d/%y %H:%M:%S", strict=False)
            .alias("Tour_date")
        )
         
        df = df.with_columns(
            pl.col('Date_STR')
            .str.strptime(pl.Datetime, format="%m/%d/%y %H:%M:%S", strict=False)
            .alias("Date")
        )

        df = df.with_columns(pl.when(pl.col('Date').is_not_null())
        .then(pl.col('Date'))
        .otherwise(pl.col('Tour_date') + pl.duration(days = pl.col('Round').map_elements(lambda x: self._round_days_lookup.get(x, None), return_dtype=pl.Int8)))
        .alias('Final_date'))

        min_date = df.select(
            pl.col("Final_date").filter(pl.col("Final_date").is_not_null()).min()
        ).item()

        return (df.with_columns(
            pl.when(pl.col("Final_date").is_not_null())
            .then(
                ((pl.col("Final_date") - min_date).dt.total_days()).cast(pl.Int64)
            )
            .otherwise(None)
            .alias("Day")
        ), min_date)
    
    def to_whr_format(self) -> List[str]:
        """
        Convert games data to WHR format.
        
        Returns:
            List of strings in WHR format: "player1_id player2_id B daynum"
        """
        games_df = self.games
        if games_df is None:
            raise ValueError("Games table not loaded")
            
        formatted_df = games_df.select(['P1', 'P2', 'Day']).with_columns([
            pl.concat_str([
                pl.col('P1').cast(pl.Int64).cast(pl.Utf8),
                pl.col('P2').cast(pl.Int64).cast(pl.Utf8),
                pl.lit('B'),
                pl.col('Day').cast(pl.Int64).cast(pl.Utf8)
            ], separator=' ').alias('whr_output')
        ])
        
        return formatted_df.filter(
            pl.col('whr_output').is_not_null()
        ).get_column('whr_output').to_list()
    
    def to_riix_format(self, sample_games = 0):
        """
        Convert games to a polars dataframe suitable for the riix library of elo, glicko, trueskill

        Returns:
        Polars dataset in format: "P1, P2, Result (1.0, or 0.0), days (integers)"
        """
        df = self.games.sort('Day')
        if sample_games != 0:
            df = df.tail(sample_games)
        df = df.with_columns(pl.lit(1.0).alias('Result'))
        return df.select(['P1', 'P2', 'Result', 'Date']).sort('Date')
    
    def to_ttt_format(self, sample_games = 0):
        """
        Convert games to a polars dataframe suitable for the TrueSkillThroughTime format

        Returns:
        Lists of games, and times they took place
        """
        df = self.games.sort('Day')
        #Allow to take a sample of the games for testing
        if sample_games != 0:
            df = df.tail(sample_games)

        games = [[[row[0]], [row[1]]] for row in df.select(['P1', 'P2']).rows()]
        days = df['Day'].to_list()
        return (games, days)
   