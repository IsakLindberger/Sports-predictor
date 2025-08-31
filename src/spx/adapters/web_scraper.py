"""Web scraping utilities for fetching live Premier League data."""

import requests
from datetime import datetime, date
from typing import List, Dict, Any, Optional
import pandas as pd
from bs4 import BeautifulSoup
import re
from loguru import logger

from ..core.schema import Match, Fixture


class PremierLeagueWebScraper:
    """Scraper for Premier League official website data."""
    
    BASE_URL = "https://www.premierleague.com"
    
    def __init__(self):
        """Initialize the scraper."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_current_season_results(self, season: str = "2025-26") -> List[Dict[str, Any]]:
        """
        Fetch current season results from Premier League website.
        
        Args:
            season: Season string (e.g., "2025-26")
            
        Returns:
            List of match dictionaries
        """
        logger.info(f"Fetching {season} season results from Premier League website")
        
        try:
            # Use the results API endpoint
            url = f"{self.BASE_URL}/results"
            response = self.session.get(url)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for match result elements
            matches = []
            
            # Find match containers (this is a simplified example - would need refinement)
            match_elements = soup.find_all('div', class_='match')
            
            for element in match_elements:
                try:
                    match_data = self._parse_match_element(element, season)
                    if match_data:
                        matches.append(match_data)
                except Exception as e:
                    logger.warning(f"Failed to parse match element: {e}")
                    continue
            
            logger.info(f"Fetched {len(matches)} matches for {season}")
            return matches
            
        except Exception as e:
            logger.error(f"Failed to fetch results: {e}")
            return []
    
    def fetch_upcoming_fixtures(self, season: str = "2025-26") -> List[Dict[str, Any]]:
        """
        Fetch upcoming fixtures from Premier League website.
        
        Args:
            season: Season string (e.g., "2025-26")
            
        Returns:
            List of fixture dictionaries
        """
        logger.info(f"Fetching {season} upcoming fixtures")
        
        try:
            url = f"{self.BASE_URL}/fixtures"
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            fixtures = []
            
            # Parse fixture elements (simplified - would need refinement)
            fixture_elements = soup.find_all('div', class_='fixture')
            
            for element in fixture_elements:
                try:
                    fixture_data = self._parse_fixture_element(element, season)
                    if fixture_data:
                        fixtures.append(fixture_data)
                except Exception as e:
                    logger.warning(f"Failed to parse fixture element: {e}")
                    continue
            
            logger.info(f"Fetched {len(fixtures)} fixtures for {season}")
            return fixtures
            
        except Exception as e:
            logger.error(f"Failed to fetch fixtures: {e}")
            return []
    
    def _parse_match_element(self, element, season: str) -> Optional[Dict[str, Any]]:
        """Parse a match result element."""
        # This is a placeholder - actual implementation would depend on 
        # the specific HTML structure of the Premier League website
        return None
    
    def _parse_fixture_element(self, element, season: str) -> Optional[Dict[str, Any]]:
        """Parse a fixture element."""
        # This is a placeholder - actual implementation would depend on 
        # the specific HTML structure of the Premier League website
        return None


class FootballDataAPI:
    """Alternative data source using football-data.org API."""
    
    BASE_URL = "https://api.football-data.org/v4"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the API client.
        
        Args:
            api_key: API key for football-data.org (free tier available)
        """
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({'X-Auth-Token': api_key})
    
    def fetch_premier_league_matches(self, season: int = 2025) -> List[Dict[str, Any]]:
        """
        Fetch Premier League matches from football-data.org API.
        
        Args:
            season: Season year (e.g., 2025 for 2025-26 season)
            
        Returns:
            List of match dictionaries
        """
        logger.info(f"Fetching {season}-{season+1} season data from football-data.org")
        
        try:
            # Premier League competition ID
            competition_id = "PL"
            url = f"{self.BASE_URL}/competitions/{competition_id}/matches"
            
            params = {
                'season': season,
                'status': 'FINISHED'  # Can also use 'SCHEDULED' for fixtures
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            matches = []
            
            for match in data.get('matches', []):
                try:
                    match_data = self._parse_api_match(match, f"{season}-{str(season+1)[-2:]}")
                    if match_data:
                        matches.append(match_data)
                except Exception as e:
                    logger.warning(f"Failed to parse match: {e}")
                    continue
            
            logger.info(f"Fetched {len(matches)} matches for {season}-{season+1}")
            return matches
            
        except Exception as e:
            logger.error(f"Failed to fetch from API: {e}")
            return []
    
    def _parse_api_match(self, match_data: Dict, season: str) -> Optional[Dict[str, Any]]:
        """Parse a match from the API response."""
        try:
            # Extract match date
            match_date = datetime.fromisoformat(
                match_data['utcDate'].replace('Z', '+00:00')
            ).date()
            
            # Extract team names
            home_team = match_data['homeTeam']['name']
            away_team = match_data['awayTeam']['name']
            
            # Extract score if available
            score = match_data.get('score', {})
            full_time = score.get('fullTime', {})
            home_goals = full_time.get('home')
            away_goals = full_time.get('away')
            
            return {
                'date': match_date,
                'season': season,
                'home_team': home_team,
                'away_team': away_team,
                'home_goals': home_goals,
                'away_goals': away_goals,
                'status': match_data.get('status')
            }
            
        except Exception as e:
            logger.warning(f"Failed to parse match data: {e}")
            return None


def create_web_data_fetcher() -> FootballDataAPI:
    """Create a data fetcher instance."""
    # Note: For production use, you'd want to add an API key
    # Free tier allows 10 requests per minute
    return FootballDataAPI()


def fetch_and_save_current_season(
    output_path: str = "data/raw/epl_2025_26.csv",
    season: int = 2025
) -> bool:
    """
    Fetch current season data and save to CSV.
    
    Args:
        output_path: Path to save the CSV file
        season: Season year
        
    Returns:
        True if successful, False otherwise
    """
    try:
        fetcher = create_web_data_fetcher()
        matches = fetcher.fetch_premier_league_matches(season)
        
        if not matches:
            logger.warning("No matches fetched")
            return False
        
        # Convert to DataFrame and save
        df = pd.DataFrame(matches)
        
        # Filter only finished matches for training data
        finished_matches = df[df['status'] == 'FINISHED'].copy()
        
        if len(finished_matches) == 0:
            logger.warning("No finished matches found")
            return False
        
        # Convert to expected CSV format
        finished_matches = finished_matches[
            ['date', 'home_team', 'away_team', 'home_goals', 'away_goals']
        ].copy()
        
        # Rename columns to match expected format
        finished_matches.columns = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
        
        # Save to CSV
        finished_matches.to_csv(output_path, index=False)
        logger.info(f"Saved {len(finished_matches)} matches to {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to fetch and save data: {e}")
        return False
