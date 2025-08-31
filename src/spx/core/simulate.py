"""Season simulation using Monte Carlo methods."""

from datetime import datetime, date as Date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from .schema import Fixture, MatchPrediction, SeasonPrediction
from .models.scoreline import BivariatePoisson


class SeasonSimulator:
    """Monte Carlo season simulation."""
    
    def __init__(
        self,
        n_simulations: int = 20000,
        random_seed: int = 42,
        use_head_to_head: bool = True
    ):
        """Initialize season simulator.
        
        Args:
            n_simulations: Number of Monte Carlo simulations
            random_seed: Random seed for reproducibility
            use_head_to_head: Whether to use head-to-head for tie-breaking
        """
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        self.use_head_to_head = use_head_to_head
        
        # Set random seed
        np.random.seed(random_seed)
    
    def simulate_season(
        self,
        fixtures: List[Fixture],
        predictions: List[MatchPrediction],
        completed_matches: Optional[List] = None
    ) -> List[SeasonPrediction]:
        """Simulate full season and generate team probabilities.
        
        Args:
            fixtures: Remaining fixtures to simulate
            predictions: Match predictions for fixtures
            completed_matches: Already completed matches (for partial season)
            
        Returns:
            List of SeasonPrediction objects for each team
        """
        logger.info(f"Running {self.n_simulations} season simulations")
        
        # Create prediction lookup
        pred_lookup = {}
        for pred in predictions:
            key = (pred.date, pred.home_team, pred.away_team)
            pred_lookup[key] = pred
        
        # Get all teams
        teams = set()
        for fixture in fixtures:
            teams.add(fixture.home_team)
            teams.add(fixture.away_team)
        teams = sorted(list(teams))
        
        # Initialize results tracking
        final_positions = {team: [] for team in teams}
        title_wins = {team: 0 for team in teams}
        top4_finishes = {team: 0 for team in teams}
        relegation_finishes = {team: 0 for team in teams}
        
        # Track cumulative statistics for average table
        cumulative_stats = {team: {
            'points': 0, 'wins': 0, 'draws': 0, 'losses': 0,
            'goals_for': 0, 'goals_against': 0, 'games_played': 0
        } for team in teams}
        
        # Run simulations
        for sim in tqdm(range(self.n_simulations), desc="Simulating seasons"):
            season_table = self._simulate_single_season(
                fixtures, pred_lookup, teams, completed_matches
            )
            
            # Record results
            for pos, team in enumerate(season_table.index, 1):
                final_positions[team].append(pos)
                
                if pos == 1:
                    title_wins[team] += 1
                if pos <= 4:
                    top4_finishes[team] += 1
                if pos >= len(teams) - 2:  # Bottom 3
                    relegation_finishes[team] += 1
            
            # Accumulate statistics for average table
            for team in teams:
                team_stats = season_table.loc[team]
                cumulative_stats[team]['points'] += team_stats['points']
                cumulative_stats[team]['wins'] += team_stats['wins']
                cumulative_stats[team]['draws'] += team_stats['draws']
                cumulative_stats[team]['losses'] += team_stats['losses']
                cumulative_stats[team]['goals_for'] += team_stats['goals_for']
                cumulative_stats[team]['goals_against'] += team_stats['goals_against']
                cumulative_stats[team]['games_played'] += team_stats['played']
        
        # Generate predictions for each team
        season_predictions = []
        
        for team in teams:
            positions = final_positions[team]
            
            # Calculate position probabilities
            position_probs = {}
            for pos in range(1, len(teams) + 1):
                position_probs[pos] = positions.count(pos) / self.n_simulations
            
            # Calculate key probabilities
            prob_title = title_wins[team] / self.n_simulations
            prob_top4 = top4_finishes[team] / self.n_simulations
            prob_relegation = relegation_finishes[team] / self.n_simulations
            
            # Expected position
            expected_position = np.mean(positions)
            
            # Expected points (simplified calculation)
            expected_points = self._estimate_expected_points(
                team, fixtures, pred_lookup, completed_matches
            )
            
            season_pred = SeasonPrediction(
                season=fixtures[0].season if fixtures else "2024-25",
                team=team,
                position_probs=position_probs,
                expected_position=expected_position,
                prob_title=prob_title,
                prob_top4=prob_top4,
                prob_relegation=prob_relegation,
                expected_points=expected_points,
                expected_goal_difference=0.0,  # Placeholder
                n_simulations=self.n_simulations,
                simulation_timestamp=datetime.now()
            )
            
            season_predictions.append(season_pred)
        
        # Calculate average table
        self.average_table = self._calculate_average_table(cumulative_stats, teams)
        
        logger.info(f"Generated season predictions for {len(teams)} teams")
        return season_predictions
    
    def _calculate_average_table(self, cumulative_stats: Dict, teams: List[str]) -> pd.DataFrame:
        """Calculate average table across all simulations.
        
        Args:
            cumulative_stats: Cumulative statistics from all simulations
            teams: List of team names
            
        Returns:
            DataFrame with average table statistics
        """
        logger.info("Calculating average table from all simulations")
        
        avg_table_data = []
        for team in teams:
            stats = cumulative_stats[team]
            avg_stats = {
                'Team': team,
                'Games_Played': stats['games_played'] / self.n_simulations,
                'Wins': stats['wins'] / self.n_simulations,
                'Draws': stats['draws'] / self.n_simulations,
                'Losses': stats['losses'] / self.n_simulations,
                'Goals_For': stats['goals_for'] / self.n_simulations,
                'Goals_Against': stats['goals_against'] / self.n_simulations,
                'Goal_Difference': (stats['goals_for'] - stats['goals_against']) / self.n_simulations,
                'Points': stats['points'] / self.n_simulations,
            }
            avg_table_data.append(avg_stats)
        
        # Create DataFrame and sort by points
        avg_table = pd.DataFrame(avg_table_data)
        avg_table = avg_table.sort_values(['Points', 'Goal_Difference', 'Goals_For'], 
                                         ascending=[False, False, False])
        avg_table = avg_table.reset_index(drop=True)
        avg_table.index = avg_table.index + 1  # Start positions from 1
        avg_table.index.name = 'Position'
        
        logger.info("Average table calculated successfully")
        return avg_table
    
    def save_average_table(self, filepath: str) -> None:
        """Save the average table to a file.
        
        Args:
            filepath: Path to save the average table
        """
        if not hasattr(self, 'average_table'):
            logger.error("Average table not calculated. Run simulate_season first.")
            return
        
        self.average_table.to_csv(filepath)
        logger.info(f"Average table saved to {filepath}")
        
        # Also log the table to console
        logger.info("Average Season Table (across {} simulations):", self.n_simulations)
        print(f"\n=== AVERAGE SEASON TABLE ({self.n_simulations:,} simulations) ===")
        print(self.average_table.round(2).to_string())
    
    def _simulate_single_season(
        self,
        fixtures: List[Fixture],
        pred_lookup: Dict[Tuple, MatchPrediction],
        teams: List[str],
        completed_matches: Optional[List] = None
    ) -> pd.DataFrame:
        """Simulate a single season.
        
        Args:
            fixtures: Fixtures to simulate
            pred_lookup: Prediction lookup dictionary
            teams: List of all teams
            completed_matches: Completed matches
            
        Returns:
            Final league table DataFrame
        """
        # Initialize table
        table = pd.DataFrame({
            'team': teams,
            'points': 0,
            'played': 0,
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'goals_for': 0,
            'goals_against': 0,
            'goal_difference': 0
        }).set_index('team')
        
        # Add completed matches if provided
        if completed_matches:
            for match in completed_matches:
                self._update_table(table, match.home_team, match.away_team,
                                 match.home_goals, match.away_goals)
        
        # Simulate remaining fixtures
        for fixture in fixtures:
            key = (fixture.date, fixture.home_team, fixture.away_team)
            
            if key in pred_lookup:
                pred = pred_lookup[key]
                
                # Sample outcome based on probabilities
                outcome_rand = np.random.random()
                
                if outcome_rand < pred.prob_home_win:
                    # Home win - sample scoreline
                    home_goals, away_goals = self._sample_winning_scoreline(
                        pred.expected_home_goals, pred.expected_away_goals, home_wins=True
                    )
                elif outcome_rand < pred.prob_home_win + pred.prob_draw:
                    # Draw - sample draw scoreline
                    home_goals, away_goals = self._sample_draw_scoreline(
                        pred.expected_home_goals, pred.expected_away_goals
                    )
                else:
                    # Away win - sample scoreline
                    home_goals, away_goals = self._sample_winning_scoreline(
                        pred.expected_home_goals, pred.expected_away_goals, home_wins=False
                    )
                
                # Update table
                self._update_table(table, fixture.home_team, fixture.away_team,
                                 home_goals, away_goals)
        
        # Sort table by points, goal difference, goals scored
        table['goal_difference'] = table['goals_for'] - table['goals_against']
        table = table.sort_values(
            ['points', 'goal_difference', 'goals_for'],
            ascending=[False, False, False]
        )
        
        return table
    
    def _update_table(
        self,
        table: pd.DataFrame,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int
    ) -> None:
        """Update league table with match result.
        
        Args:
            table: League table DataFrame
            home_team: Home team name
            away_team: Away team name
            home_goals: Home team goals
            away_goals: Away team goals
        """
        # Update goals
        table.loc[home_team, 'goals_for'] += home_goals
        table.loc[home_team, 'goals_against'] += away_goals
        table.loc[away_team, 'goals_for'] += away_goals
        table.loc[away_team, 'goals_against'] += home_goals
        
        # Update games played
        table.loc[home_team, 'played'] += 1
        table.loc[away_team, 'played'] += 1
        
        # Update results
        if home_goals > away_goals:
            # Home win
            table.loc[home_team, 'wins'] += 1
            table.loc[home_team, 'points'] += 3
            table.loc[away_team, 'losses'] += 1
        elif home_goals < away_goals:
            # Away win
            table.loc[away_team, 'wins'] += 1
            table.loc[away_team, 'points'] += 3
            table.loc[home_team, 'losses'] += 1
        else:
            # Draw
            table.loc[home_team, 'draws'] += 1
            table.loc[home_team, 'points'] += 1
            table.loc[away_team, 'draws'] += 1
            table.loc[away_team, 'points'] += 1
    
    def _sample_winning_scoreline(
        self,
        exp_home: float,
        exp_away: float,
        home_wins: bool
    ) -> Tuple[int, int]:
        """Sample a scoreline where one team wins.
        
        Args:
            exp_home: Expected home goals
            exp_away: Expected away goals
            home_wins: Whether home team wins
            
        Returns:
            Tuple of (home_goals, away_goals)
        """
        # Simple sampling - can be enhanced with actual bivariate Poisson
        if home_wins:
            home_goals = max(1, int(np.random.poisson(exp_home * 1.2)))
            away_goals = int(np.random.poisson(exp_away * 0.8))
            if away_goals >= home_goals:
                away_goals = max(0, home_goals - 1)
        else:
            away_goals = max(1, int(np.random.poisson(exp_away * 1.2)))
            home_goals = int(np.random.poisson(exp_home * 0.8))
            if home_goals >= away_goals:
                home_goals = max(0, away_goals - 1)
        
        return home_goals, away_goals
    
    def _sample_draw_scoreline(
        self,
        exp_home: float,
        exp_away: float
    ) -> Tuple[int, int]:
        """Sample a draw scoreline.
        
        Args:
            exp_home: Expected home goals
            exp_away: Expected away goals
            
        Returns:
            Tuple of (home_goals, away_goals) where both are equal
        """
        # Sample goals and force draw
        avg_goals = (exp_home + exp_away) / 2
        goals = int(np.random.poisson(avg_goals))
        return goals, goals
    
    def _estimate_expected_points(
        self,
        team: str,
        fixtures: List[Fixture],
        pred_lookup: Dict[Tuple, MatchPrediction],
        completed_matches: Optional[List] = None
    ) -> float:
        """Estimate expected points for a team.
        
        Args:
            team: Team name
            fixtures: Remaining fixtures
            pred_lookup: Prediction lookup
            completed_matches: Completed matches
            
        Returns:
            Expected points for the season
        """
        expected_points = 0.0
        
        # Add points from completed matches
        if completed_matches:
            for match in completed_matches:
                if match.home_team == team:
                    if match.outcome == 'H':
                        expected_points += 3
                    elif match.outcome == 'D':
                        expected_points += 1
                elif match.away_team == team:
                    if match.outcome == 'A':
                        expected_points += 3
                    elif match.outcome == 'D':
                        expected_points += 1
        
        # Add expected points from remaining fixtures
        for fixture in fixtures:
            key = (fixture.date, fixture.home_team, fixture.away_team)
            
            if key in pred_lookup:
                pred = pred_lookup[key]
                
                if fixture.home_team == team:
                    expected_points += pred.prob_home_win * 3 + pred.prob_draw * 1
                elif fixture.away_team == team:
                    expected_points += pred.prob_away_win * 3 + pred.prob_draw * 1
        
        return expected_points
