"""Elo rating system for team strength estimation."""

from datetime import date as Date
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from .schema import Match, TeamRating


class EloRatingSystem:
    """Elo rating system for football teams."""
    
    def __init__(
        self,
        k_factor: float = 20.0,
        base_rating: float = 1500.0,
        home_advantage: float = 30.0,
        margin_multiplier: float = 0.1
    ):
        """Initialize Elo rating system.
        
        Args:
            k_factor: Elo K-factor (rating change scale)
            base_rating: Starting rating for new teams
            home_advantage: Home team rating boost
            margin_multiplier: Goal margin impact on rating change
        """
        self.k_factor = k_factor
        self.base_rating = base_rating
        self.home_advantage = home_advantage
        self.margin_multiplier = margin_multiplier
        
        # Track current ratings
        self.ratings: Dict[str, float] = {}
        self.games_played: Dict[str, int] = {}
        self.rating_history: Dict[str, Dict[Date, float]] = {}
    
    def update_ratings(self, matches: List[Match]) -> List[TeamRating]:
        """Update Elo ratings based on match results.
        
        Args:
            matches: List of matches in chronological order
            
        Returns:
            List of TeamRating objects with rating history
        """
        logger.info(f"Updating Elo ratings for {len(matches)} matches")
        
        all_ratings = []
        
        for match in matches:
            # Initialize teams if not seen before
            self._initialize_team(match.home_team)
            self._initialize_team(match.away_team)
            
            # Get pre-match ratings
            home_rating = self.ratings[match.home_team]
            away_rating = self.ratings[match.away_team]
            
            # Calculate rating changes
            home_change, away_change = self._calculate_rating_changes(
                home_rating, away_rating, match
            )
            
            # Update ratings
            self.ratings[match.home_team] += home_change
            self.ratings[match.away_team] += away_change
            
            # Update games played
            self.games_played[match.home_team] += 1
            self.games_played[match.away_team] += 1
            
            # Store rating history
            self.rating_history[match.home_team][match.date] = self.ratings[match.home_team]
            self.rating_history[match.away_team][match.date] = self.ratings[match.away_team]
            
            # Create rating objects
            home_rating_obj = TeamRating(
                team=match.home_team,
                date=match.date,
                elo_rating=self.ratings[match.home_team],
                games_played=self.games_played[match.home_team]
            )
            
            away_rating_obj = TeamRating(
                team=match.away_team,
                date=match.date,
                elo_rating=self.ratings[match.away_team],
                games_played=self.games_played[match.away_team]
            )
            
            all_ratings.extend([home_rating_obj, away_rating_obj])
        
        logger.info(f"Generated {len(all_ratings)} rating records")
        return all_ratings
    
    def _initialize_team(self, team: str) -> None:
        """Initialize a new team with base rating.
        
        Args:
            team: Team name
        """
        if team not in self.ratings:
            self.ratings[team] = self.base_rating
            self.games_played[team] = 0
            self.rating_history[team] = {}
    
    def _calculate_rating_changes(
        self, 
        home_rating: float, 
        away_rating: float, 
        match: Match
    ) -> Tuple[float, float]:
        """Calculate Elo rating changes for both teams.
        
        Args:
            home_rating: Home team's current rating
            away_rating: Away team's current rating
            match: Match result
            
        Returns:
            Tuple of (home_change, away_change)
        """
        # Adjust for home advantage
        effective_home_rating = home_rating + self.home_advantage
        
        # Calculate expected scores
        expected_home = 1 / (1 + 10**((away_rating - effective_home_rating) / 400))
        expected_away = 1 - expected_home
        
        # Determine actual scores
        if match.outcome == 'H':
            actual_home = 1.0
            actual_away = 0.0
        elif match.outcome == 'A':
            actual_home = 0.0
            actual_away = 1.0
        else:  # Draw
            actual_home = 0.5
            actual_away = 0.5
        
        # Calculate goal margin multiplier
        goal_margin = abs(match.home_goals - match.away_goals)
        margin_factor = 1 + self.margin_multiplier * goal_margin
        
        # Calculate rating changes
        home_change = self.k_factor * margin_factor * (actual_home - expected_home)
        away_change = self.k_factor * margin_factor * (actual_away - expected_away)
        
        return home_change, away_change
    
    def get_current_ratings(self) -> Dict[str, float]:
        """Get current ratings for all teams.
        
        Returns:
            Dictionary mapping team names to current ratings
        """
        return self.ratings.copy()
    
    def get_rating_history(self) -> Dict[str, Dict[Date, float]]:
        """Get full rating history for all teams.
        
        Returns:
            Dictionary mapping team names to their rating history
        """
        return self.rating_history.copy()
    
    def predict_match_outcome(
        self, 
        home_team: str, 
        away_team: str
    ) -> Tuple[float, float, float]:
        """Predict match outcome probabilities using Elo ratings.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            
        Returns:
            Tuple of (home_win_prob, draw_prob, away_win_prob)
        """
        # Get current ratings
        home_rating = self.ratings.get(home_team, self.base_rating)
        away_rating = self.ratings.get(away_team, self.base_rating)
        
        # Adjust for home advantage
        effective_home_rating = home_rating + self.home_advantage
        
        # Calculate win expectancy
        home_win_expectancy = 1 / (1 + 10**((away_rating - effective_home_rating) / 400))
        away_win_expectancy = 1 - home_win_expectancy
        
        # Convert to match outcome probabilities (simplified model)
        # This is a basic conversion - the actual Dixon-Coles model will be more sophisticated
        draw_prob = 0.27  # Average draw probability in football
        home_win_prob = home_win_expectancy * (1 - draw_prob)
        away_win_prob = away_win_expectancy * (1 - draw_prob)
        
        # Normalize to ensure probabilities sum to 1
        total = home_win_prob + draw_prob + away_win_prob
        home_win_prob /= total
        draw_prob /= total
        away_win_prob /= total
        
        return home_win_prob, draw_prob, away_win_prob
