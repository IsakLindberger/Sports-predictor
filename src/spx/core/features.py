"""Feature engineering for match prediction."""

from datetime import date as Date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .schema import Match, TeamRating


class FeatureEngineer:
    """Feature engineering for football matches."""
    
    def __init__(self, lookback_games: int = 6, min_games_for_rating: int = 10):
        """Initialize feature engineer.
        
        Args:
            lookback_games: Number of recent games for rolling features
            min_games_for_rating: Minimum games before using Elo ratings
        """
        self.lookback_games = lookback_games
        self.min_games_for_rating = min_games_for_rating
    
    def create_features(
        self,
        matches: List[Match],
        ratings: Optional[Dict[str, Dict[Date, float]]] = None
    ) -> pd.DataFrame:
        """Create features for all matches.
        
        Args:
            matches: List of matches
            ratings: Optional Elo ratings dict {team: {date: rating}}
            
        Returns:
            DataFrame with features for each match
        """
        logger.info(f"Creating features for {len(matches)} matches")
        
        # Handle empty matches
        if not matches:
            return pd.DataFrame()
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame([match.dict() for match in matches])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Initialize feature columns
        feature_cols = []
        
        # Add basic features
        df['home_advantage'] = 1  # Home team indicator
        feature_cols.append('home_advantage')
        
        # Add rolling form features
        form_features = self._add_form_features(df)
        feature_cols.extend(form_features)
        
        # Add rest days features
        rest_features = self._add_rest_days_features(df)
        feature_cols.extend(rest_features)
        
        # Add Elo rating features if provided
        if ratings:
            elo_features = self._add_elo_features(df, ratings)
            feature_cols.extend(elo_features)
        
        # Add head-to-head features
        h2h_features = self._add_head_to_head_features(df)
        feature_cols.extend(h2h_features)
        
        logger.info(f"Created {len(feature_cols)} features: {feature_cols}")
        
        # Return only feature columns plus target
        target_cols = ['outcome', 'home_goals', 'away_goals']
        return df[feature_cols + target_cols + ['date', 'home_team', 'away_team']]
    
    def _add_form_features(self, df: pd.DataFrame) -> List[str]:
        """Add rolling form features.
        
        Args:
            df: Matches DataFrame
            
        Returns:
            List of added feature column names
        """
        feature_cols = []
        
        # Get all unique teams
        teams = set(df['home_team'].unique()) | set(df['away_team'].unique())
        
        for i, row in df.iterrows():
            match_date = row['date']
            home_team = row['home_team']
            away_team = row['away_team']
            
            # Get recent form for each team
            home_form = self._get_team_form(df, home_team, match_date, i)
            away_form = self._get_team_form(df, away_team, match_date, i)
            
            # Home team form features
            df.loc[i, 'home_form_points'] = home_form['points']
            df.loc[i, 'home_form_gf'] = home_form['goals_for']
            df.loc[i, 'home_form_ga'] = home_form['goals_against']
            df.loc[i, 'home_form_gd'] = home_form['goal_diff']
            df.loc[i, 'home_form_games'] = home_form['games']
            
            # Away team form features
            df.loc[i, 'away_form_points'] = away_form['points']
            df.loc[i, 'away_form_gf'] = away_form['goals_for']
            df.loc[i, 'away_form_ga'] = away_form['goals_against']
            df.loc[i, 'away_form_gd'] = away_form['goal_diff']
            df.loc[i, 'away_form_games'] = away_form['games']
        
        form_features = [
            'home_form_points', 'home_form_gf', 'home_form_ga', 'home_form_gd', 'home_form_games',
            'away_form_points', 'away_form_gf', 'away_form_ga', 'away_form_gd', 'away_form_games'
        ]
        
        # Fill NaN values with 0 for teams with insufficient history
        for col in form_features:
            df[col] = df[col].fillna(0)
        
        feature_cols.extend(form_features)
        return feature_cols
    
    def _get_team_form(
        self, 
        df: pd.DataFrame, 
        team: str, 
        match_date: pd.Timestamp, 
        current_idx: int
    ) -> Dict[str, float]:
        """Get recent form statistics for a team.
        
        Args:
            df: Matches DataFrame
            team: Team name
            match_date: Date of current match
            current_idx: Index of current match (to avoid data leakage)
            
        Returns:
            Dictionary with form statistics
        """
        # Get team's recent matches (before current match)
        team_matches = df[
            (df.index < current_idx) & 
            ((df['home_team'] == team) | (df['away_team'] == team))
        ].tail(self.lookback_games)
        
        if len(team_matches) == 0:
            return {
                'points': 0.0, 'goals_for': 0.0, 'goals_against': 0.0,
                'goal_diff': 0.0, 'games': 0
            }
        
        points = 0
        goals_for = 0
        goals_against = 0
        
        for _, match in team_matches.iterrows():
            if match['home_team'] == team:
                # Team played at home
                goals_for += match['home_goals']
                goals_against += match['away_goals']
                
                if match['outcome'] == 'H':
                    points += 3
                elif match['outcome'] == 'D':
                    points += 1
            else:
                # Team played away
                goals_for += match['away_goals']
                goals_against += match['home_goals']
                
                if match['outcome'] == 'A':
                    points += 3
                elif match['outcome'] == 'D':
                    points += 1
        
        n_games = len(team_matches)
        return {
            'points': points / n_games if n_games > 0 else 0.0,
            'goals_for': goals_for / n_games if n_games > 0 else 0.0,
            'goals_against': goals_against / n_games if n_games > 0 else 0.0,
            'goal_diff': (goals_for - goals_against) / n_games if n_games > 0 else 0.0,
            'games': n_games
        }
    
    def _add_rest_days_features(self, df: pd.DataFrame) -> List[str]:
        """Add rest days features.
        
        Args:
            df: Matches DataFrame
            
        Returns:
            List of added feature column names
        """
        feature_cols = []
        
        for i, row in df.iterrows():
            match_date = row['date']
            home_team = row['home_team']
            away_team = row['away_team']
            
            # Find each team's previous match
            home_prev = df[
                (df.index < i) & 
                ((df['home_team'] == home_team) | (df['away_team'] == home_team))
            ]
            
            away_prev = df[
                (df.index < i) & 
                ((df['home_team'] == away_team) | (df['away_team'] == away_team))
            ]
            
            # Calculate rest days
            if len(home_prev) > 0:
                home_rest = (match_date - home_prev.iloc[-1]['date']).days
            else:
                home_rest = 14  # Default for first match
                
            if len(away_prev) > 0:
                away_rest = (match_date - away_prev.iloc[-1]['date']).days
            else:
                away_rest = 14  # Default for first match
            
            df.loc[i, 'home_rest_days'] = home_rest
            df.loc[i, 'away_rest_days'] = away_rest
        
        rest_features = ['home_rest_days', 'away_rest_days']
        feature_cols.extend(rest_features)
        return feature_cols
    
    def _add_elo_features(
        self, 
        df: pd.DataFrame, 
        ratings: Dict[str, Dict[Date, float]]
    ) -> List[str]:
        """Add Elo rating features.
        
        Args:
            df: Matches DataFrame
            ratings: Elo ratings dict
            
        Returns:
            List of added feature column names
        """
        feature_cols = []
        
        for i, row in df.iterrows():
            match_date = row['date'].date()
            home_team = row['home_team']
            away_team = row['away_team']
            
            # Get ratings closest to match date
            home_elo = self._get_rating_at_date(ratings, home_team, match_date)
            away_elo = self._get_rating_at_date(ratings, away_team, match_date)
            
            df.loc[i, 'home_elo'] = home_elo
            df.loc[i, 'away_elo'] = away_elo
            df.loc[i, 'elo_diff'] = home_elo - away_elo
        
        elo_features = ['home_elo', 'away_elo', 'elo_diff']
        feature_cols.extend(elo_features)
        return elo_features
    
    def _get_rating_at_date(
        self, 
        ratings: Dict[str, Dict[Date, float]], 
        team: str, 
        match_date: Date
    ) -> float:
        """Get team's Elo rating at a specific date.
        
        Args:
            ratings: Ratings dictionary
            team: Team name
            match_date: Date to get rating for
            
        Returns:
            Elo rating (1500.0 if no rating available)
        """
        if team not in ratings:
            return 1500.0  # Default rating
        
        team_ratings = ratings[team]
        
        # Find the most recent rating before or on the match date
        valid_dates = [d for d in team_ratings.keys() if d <= match_date]
        
        if not valid_dates:
            return 1500.0  # Default rating
        
        latest_date = max(valid_dates)
        return team_ratings[latest_date]
    
    def _add_head_to_head_features(self, df: pd.DataFrame) -> List[str]:
        """Add head-to-head features.
        
        Args:
            df: Matches DataFrame
            
        Returns:
            List of added feature column names
        """
        feature_cols = []
        
        for i, row in df.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            
            # Get previous meetings between these teams
            h2h_matches = df[
                (df.index < i) & 
                (
                    ((df['home_team'] == home_team) & (df['away_team'] == away_team)) |
                    ((df['home_team'] == away_team) & (df['away_team'] == home_team))
                )
            ].tail(6)  # Last 6 meetings
            
            if len(h2h_matches) == 0:
                df.loc[i, 'h2h_home_wins'] = 0
                df.loc[i, 'h2h_draws'] = 0
                df.loc[i, 'h2h_away_wins'] = 0
                df.loc[i, 'h2h_games'] = 0
            else:
                home_wins = 0
                draws = 0
                away_wins = 0
                
                for _, h2h_match in h2h_matches.iterrows():
                    if h2h_match['home_team'] == home_team:
                        # Current home team was home in H2H
                        if h2h_match['outcome'] == 'H':
                            home_wins += 1
                        elif h2h_match['outcome'] == 'D':
                            draws += 1
                        else:
                            away_wins += 1
                    else:
                        # Current home team was away in H2H
                        if h2h_match['outcome'] == 'A':
                            home_wins += 1
                        elif h2h_match['outcome'] == 'D':
                            draws += 1
                        else:
                            away_wins += 1
                
                df.loc[i, 'h2h_home_wins'] = home_wins
                df.loc[i, 'h2h_draws'] = draws
                df.loc[i, 'h2h_away_wins'] = away_wins
                df.loc[i, 'h2h_games'] = len(h2h_matches)
        
        h2h_features = ['h2h_home_wins', 'h2h_draws', 'h2h_away_wins', 'h2h_games']
        feature_cols.extend(h2h_features)
        return h2h_features


def create_target_variables(matches: List[Match]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create target variables for model training.
    
    Args:
        matches: List of matches
        
    Returns:
        Tuple of (outcome_targets, home_goals, away_goals)
    """
    outcomes = []
    home_goals = []
    away_goals = []
    
    for match in matches:
        # Outcome targets (0=away win, 1=draw, 2=home win)
        if match.outcome == 'H':
            outcomes.append(2)
        elif match.outcome == 'D':
            outcomes.append(1)
        else:  # 'A'
            outcomes.append(0)
        
        home_goals.append(match.home_goals)
        away_goals.append(match.away_goals)
    
    return (
        np.array(outcomes),
        np.array(home_goals),
        np.array(away_goals)
    )


def prepare_prediction_features(
    home_team: str,
    away_team: str,
    match_date: Date,
    historical_matches: List[Match],
    ratings: Optional[Dict[str, Dict[Date, float]]] = None,
    lookback_games: int = 6
) -> Dict[str, float]:
    """Prepare features for a single match prediction.
    
    Args:
        home_team: Home team name
        away_team: Away team name
        match_date: Match date
        historical_matches: Historical match data
        ratings: Optional Elo ratings
        lookback_games: Number of games for form calculation
        
    Returns:
        Dictionary of features
    """
    engineer = FeatureEngineer(lookback_games=lookback_games)
    
    # Create dummy match for feature extraction
    dummy_match = Match(
        date=match_date,
        season="2024-25",  # This will be overridden by actual season
        home_team=home_team,
        away_team=away_team,
        home_goals=0,  # Dummy values
        away_goals=0,
        outcome='D'
    )
    
    # Add to historical data temporarily
    all_matches = historical_matches + [dummy_match]
    
    # Create features
    features_df = engineer.create_features(all_matches, ratings)
    
    # Return features for the last row (our dummy match)
    feature_row = features_df.iloc[-1]
    
    # Remove target columns and metadata
    feature_dict = feature_row.drop(['outcome', 'home_goals', 'away_goals', 
                                   'date', 'home_team', 'away_team']).to_dict()
    
    return feature_dict
