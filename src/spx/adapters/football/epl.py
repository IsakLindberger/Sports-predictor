"""English Premier League adapter for football-data.co.uk CSV format."""

from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
from loguru import logger

from ...core.schema import Fixture, Match, MatchOutcome
from ..base import BaseAdapter


class EPLAdapter(BaseAdapter):
    """Adapter for English Premier League data from football-data.co.uk."""
    
    def __init__(self, data_dir: Path):
        """Initialize EPL adapter.
        
        Args:
            data_dir: Directory containing EPL CSV files
        """
        super().__init__(data_dir)
        
        # Column mapping from football-data.co.uk format
        self.column_mapping = {
            'Date': 'date',
            'HomeTeam': 'home_team',
            'AwayTeam': 'away_team', 
            'FTHG': 'home_goals',
            'FTAG': 'away_goals',
            'FTR': 'outcome',
            'HTHG': 'home_ht_goals',
            'HTAG': 'away_ht_goals',
            'HTR': 'ht_outcome',
            'HS': 'home_shots',
            'AS': 'away_shots',
            'HST': 'home_shots_on_target',
            'AST': 'away_shots_on_target',
            'HF': 'home_fouls',
            'AF': 'away_fouls',
            'HC': 'home_corners',
            'AC': 'away_corners',
            'HY': 'home_yellow_cards',
            'AY': 'away_yellow_cards',
            'HR': 'home_red_cards',
            'AR': 'away_red_cards'
        }
    
    def load_matches(self, season: str) -> List[Match]:
        """Load EPL matches for a season.
        
        Args:
            season: Season in format '2023-24'
            
        Returns:
            List of Match objects
        """
        filepath = self._get_season_file(season)
        
        if not filepath.exists():
            raise FileNotFoundError(f"EPL data file not found: {filepath}")
        
        logger.info(f"Loading EPL matches from {filepath}")
        
        # Load CSV
        df = pd.read_csv(filepath)
        
        # Validate format
        if not self.validate_data_format(filepath):
            raise ValueError(f"Invalid data format in {filepath}")
        
        # Rename columns
        df = df.rename(columns=self.column_mapping)
        
        # Filter out matches without results (upcoming fixtures)
        df = df.dropna(subset=['home_goals', 'away_goals', 'outcome'])
        
        # Convert dates
        df['date'] = pd.to_datetime(df['date'], dayfirst=True).dt.date
        
        # Create Match objects
        matches = []
        for _, row in df.iterrows():
            match_data = {
                'date': row['date'],
                'season': season,
                'home_team': self._normalize_team_name(row['home_team']),
                'away_team': self._normalize_team_name(row['away_team']),
                'home_goals': int(row['home_goals']),
                'away_goals': int(row['away_goals']),
                'outcome': MatchOutcome(row['outcome'])
            }
            
            # Add optional fields if available
            optional_fields = [
                'home_shots', 'away_shots', 'home_shots_on_target', 'away_shots_on_target',
                'home_red_cards', 'away_red_cards', 'home_yellow_cards', 'away_yellow_cards'
            ]
            
            for field in optional_fields:
                if field in row and pd.notna(row[field]):
                    match_data[field] = int(row[field])
            
            matches.append(Match(**match_data))
        
        logger.info(f"Loaded {len(matches)} EPL matches for season {season}")
        return matches
    
    def load_fixtures(self, season: str) -> List[Fixture]:
        """Load EPL fixtures for a season.
        
        Args:
            season: Season in format '2023-24'
            
        Returns:
            List of Fixture objects
        """
        filepath = self._get_season_file(season)
        
        if not filepath.exists():
            logger.warning(f"EPL fixtures file not found: {filepath}")
            return []
        
        logger.info(f"Loading EPL fixtures from {filepath}")
        
        # Load CSV
        df = pd.read_csv(filepath)
        df = df.rename(columns=self.column_mapping)
        
        # Filter to only future fixtures (no result yet)
        future_fixtures = df[df['outcome'].isna() | (df['home_goals'].isna())]
        
        if len(future_fixtures) == 0:
            logger.info("No future fixtures found")
            return []
        
        # Convert dates
        future_fixtures['date'] = pd.to_datetime(future_fixtures['date'], dayfirst=True).dt.date
        
        # Create Fixture objects
        fixtures = []
        for _, row in future_fixtures.iterrows():
            fixture = Fixture(
                date=row['date'],
                season=season,
                home_team=self._normalize_team_name(row['home_team']),
                away_team=self._normalize_team_name(row['away_team'])
            )
            fixtures.append(fixture)
        
        logger.info(f"Loaded {len(fixtures)} EPL fixtures for season {season}")
        return fixtures
    
    def get_available_seasons(self) -> List[str]:
        """Get list of available EPL seasons.
        
        Returns:
            List of season identifiers
        """
        seasons = []
        
        # Look for CSV files in data directory
        for csv_file in self.data_dir.glob("*.csv"):
            # Try to extract season from filename
            # Expected formats: E0.csv, epl_2023_24.csv, etc.
            filename = csv_file.stem.lower()
            
            if 'epl' in filename and any(char.isdigit() for char in filename):
                # Extract year pattern
                parts = filename.split('_')
                if len(parts) >= 3:
                    try:
                        year1 = int(parts[-2])
                        year2 = int(parts[-1])
                        if year2 == year1 + 1 or year2 == (year1 + 1) % 100:
                            season = f"{year1}-{year2:02d}"
                            seasons.append(season)
                    except (ValueError, IndexError):
                        pass
            elif filename == 'e0':  # Current season file
                current_year = datetime.now().year
                if datetime.now().month >= 8:  # Season starts in August
                    season = f"{current_year}-{(current_year + 1) % 100:02d}"
                else:
                    season = f"{current_year - 1}-{current_year % 100:02d}"
                seasons.append(season)
        
        return sorted(list(set(seasons)))
    
    def validate_data_format(self, filepath: Path) -> bool:
        """Validate EPL CSV format.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            True if format is valid
        """
        try:
            df = pd.read_csv(filepath, nrows=1)
            
            # Check for required columns
            required_cols = ['Date', 'HomeTeam', 'AwayTeam']
            
            for col in required_cols:
                if col not in df.columns:
                    logger.error(f"Required column '{col}' not found in {filepath}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating {filepath}: {e}")
            return False
    
    def _get_season_file(self, season: str) -> Path:
        """Get filepath for a season.
        
        Args:
            season: Season identifier
            
        Returns:
            Path to season data file
        """
        # Try different filename patterns
        season_clean = season.replace('-', '_')
        
        possible_files = [
            f"epl_{season_clean}.csv",
            f"EPL_{season_clean}.csv",
            f"E0_{season_clean}.csv",
            "E0.csv",  # Current season
        ]
        
        for filename in possible_files:
            filepath = self.data_dir / filename
            if filepath.exists():
                return filepath
        
        # Return most likely filename even if not found
        return self.data_dir / f"epl_{season_clean}.csv"
    
    def _normalize_team_name(self, team_name: str) -> str:
        """Normalize team names for consistency.
        
        Args:
            team_name: Raw team name
            
        Returns:
            Normalized team name
        """
        # EPL team name mappings for consistency
        name_mappings = {
            'Man City': 'Manchester City',
            'Man United': 'Manchester United',
            'Man Utd': 'Manchester United',
            'Spurs': 'Tottenham',
            'Tottenham Hotspur': 'Tottenham',
            'Newcastle': 'Newcastle United',
            'West Ham': 'West Ham United',
            'Wolves': 'Wolverhampton Wanderers',
            'Nott\'m Forest': 'Nottingham Forest',
            'Sheffield United': 'Sheffield Utd',
            'Brighton': 'Brighton & Hove Albion',
            'Leicester': 'Leicester City',
            'Norwich': 'Norwich City',
            'Crystal Palace': 'Crystal Palace',
        }
        
        return name_mappings.get(team_name, team_name)
