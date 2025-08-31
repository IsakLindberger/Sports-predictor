import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import combinations
import random
from tqdm import tqdm
import json

def generate_full_season_fixtures():
    """Generate all 380 fixtures for a Premier League season"""
    
    # All 20 Premier League teams for 2025-26
    teams = [
        "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton & Hove Albion",
        "Burnley", "Chelsea", "Crystal Palace", "Everton", "Fulham",
        "Leeds United", "Liverpool", "Manchester City", "Manchester United", 
        "Newcastle United", "Nottingham Forest", "Southampton", "Tottenham", 
        "West Ham United", "Wolverhampton Wanderers"
    ]
    
    fixtures = []
    start_date = datetime(2025, 8, 16)  # Traditional PL season start
    current_date = start_date
    
    # Generate round-robin fixtures
    all_matchups = list(combinations(teams, 2))
    
    # Create home and away fixtures
    round_fixtures = []
    for home, away in all_matchups:
        round_fixtures.append((home, away))
        round_fixtures.append((away, home))
    
    random.shuffle(round_fixtures)
    
    # Distribute across 38 gameweeks
    for round_num in range(38):
        round_start = round_num * 10
        round_end = (round_num + 1) * 10
        round_matches = round_fixtures[round_start:round_end]
        
        for i, (home, away) in enumerate(round_matches):
            match_date = current_date if i < 6 else current_date + timedelta(days=1)
            
            fixtures.append({
                'date': match_date,
                'home_team': home,
                'away_team': away,
                'season': '2025-26',
                'gameweek': round_num + 1
            })
        
        current_date += timedelta(days=7)
    
    return pd.DataFrame(fixtures)

def calculate_team_stats(matches_df, team_name):
    """Calculate team statistics from historical data"""
    home_matches = matches_df[matches_df['home_team'] == team_name]
    away_matches = matches_df[matches_df['away_team'] == team_name]
    
    goals_for = home_matches['home_goals'].sum() + away_matches['away_goals'].sum()
    goals_against = home_matches['away_goals'].sum() + away_matches['home_goals'].sum()
    
    home_points = (home_matches['home_goals'] > home_matches['away_goals']).sum() * 3 + \
                  (home_matches['home_goals'] == home_matches['away_goals']).sum() * 1
    away_points = (away_matches['away_goals'] > away_matches['home_goals']).sum() * 3 + \
                  (away_matches['away_goals'] == away_matches['home_goals']).sum() * 1
    
    total_matches = len(home_matches) + len(away_matches)
    
    return {
        'matches': total_matches,
        'goals_for': goals_for,
        'goals_against': goals_against,
        'goal_avg_for': goals_for / max(total_matches, 1),
        'goal_avg_against': goals_against / max(total_matches, 1),
        'points': home_points + away_points,
        'ppg': (home_points + away_points) / max(total_matches, 1)
    }

def simulate_match(home_team, away_team, team_stats, league_avg_goals, home_advantage):
    """Simulate a single match result"""
    
    home_stats = team_stats.get(home_team, {
        'goal_avg_for': league_avg_goals/2, 'goal_avg_against': league_avg_goals/2, 'ppg': 1.0
    })
    away_stats = team_stats.get(away_team, {
        'goal_avg_for': league_avg_goals/2, 'goal_avg_against': league_avg_goals/2, 'ppg': 1.0
    })
    
    # Calculate expected goals with home advantage
    base_home_goals = (home_stats['goal_avg_for'] + away_stats['goal_avg_against']) / 2 + home_advantage/2
    base_away_goals = (away_stats['goal_avg_for'] + home_stats['goal_avg_against']) / 2 - home_advantage/2
    
    # Use Poisson distribution for goal generation
    home_goals = max(0, np.random.poisson(max(0.1, base_home_goals)))
    away_goals = max(0, np.random.poisson(max(0.1, base_away_goals)))
    
    # Calculate points
    if home_goals > away_goals:
        return home_goals, away_goals, 3, 0
    elif away_goals > home_goals:
        return home_goals, away_goals, 0, 3
    else:
        return home_goals, away_goals, 1, 1

def simulate_season(fixtures_df, team_stats, league_avg_goals, home_advantage):
    """Simulate a complete season"""
    
    teams = set(fixtures_df['home_team'].unique()) | set(fixtures_df['away_team'].unique())
    table = {team: {
        'points': 0, 'played': 0, 'won': 0, 'drawn': 0, 'lost': 0,
        'goals_for': 0, 'goals_against': 0, 'goal_difference': 0
    } for team in teams}
    
    # Simulate all matches
    for _, fixture in fixtures_df.iterrows():
        home_team = fixture['home_team']
        away_team = fixture['away_team']
        
        home_goals, away_goals, home_points, away_points = simulate_match(
            home_team, away_team, team_stats, league_avg_goals, home_advantage
        )
        
        # Update home team stats
        table[home_team]['points'] += home_points
        table[home_team]['played'] += 1
        table[home_team]['goals_for'] += home_goals
        table[home_team]['goals_against'] += away_goals
        table[home_team]['goal_difference'] = table[home_team]['goals_for'] - table[home_team]['goals_against']
        
        if home_goals > away_goals:
            table[home_team]['won'] += 1
        elif home_goals == away_goals:
            table[home_team]['drawn'] += 1
        else:
            table[home_team]['lost'] += 1
        
        # Update away team stats
        table[away_team]['points'] += away_points
        table[away_team]['played'] += 1
        table[away_team]['goals_for'] += away_goals
        table[away_team]['goals_against'] += home_goals
        table[away_team]['goal_difference'] = table[away_team]['goals_for'] - table[away_team]['goals_against']
        
        if away_goals > home_goals:
            table[away_team]['won'] += 1
        elif away_goals == home_goals:
            table[away_team]['drawn'] += 1
        else:
            table[away_team]['lost'] += 1
    
    return table

def main():
    print("🏆 Premier League 2025-26 Season Simulation (20,000 iterations)")
    print("=" * 65)
    
    # Load historical data
    matches_df = pd.read_parquet('data/processed/matches.parquet')
    
    # Generate full season fixtures
    print("🔄 Generating complete 2025-26 season fixture list...")
    fixtures_df = generate_full_season_fixtures()
    print(f"✅ Generated {len(fixtures_df)} fixtures across 38 gameweeks")
    
    # Calculate team statistics
    all_teams = set(fixtures_df['home_team'].unique()) | set(fixtures_df['away_team'].unique())
    team_stats = {team: calculate_team_stats(matches_df, team) for team in all_teams}
    
    # League statistics
    league_avg_goals = matches_df['home_goals'].mean() + matches_df['away_goals'].mean()
    home_advantage = matches_df['home_goals'].mean() - matches_df['away_goals'].mean()
    
    print(f"\n📊 Based on {len(matches_df)} historical matches:")
    print(f"   Average goals per match: {league_avg_goals:.1f}")
    print(f"   Home advantage: +{home_advantage:.2f} goals")
    
    # Run simulations
    num_simulations = 20000
    print(f"\n🎲 Running {num_simulations:,} season simulations...")
    
    # Initialize aggregation
    team_results = {team: {
        'total_points': 0, 'total_gf': 0, 'total_ga': 0, 'total_gd': 0,
        'total_wins': 0, 'total_draws': 0, 'total_losses': 0,
        'position_counts': [0] * 20,
        'title_wins': 0, 'top4_finishes': 0, 'relegations': 0
    } for team in all_teams}
    
    # Run simulations
    for sim in tqdm(range(num_simulations), desc="Simulating seasons"):
        season_table = simulate_season(fixtures_df, team_stats, league_avg_goals, home_advantage)
        
        # Sort teams by Premier League rules
        sorted_teams = sorted(season_table.items(), 
                            key=lambda x: (x[1]['points'], x[1]['goal_difference'], x[1]['goals_for']), 
                            reverse=True)
        
        # Update position tracking
        for pos, (team, stats) in enumerate(sorted_teams):
            team_results[team]['total_points'] += stats['points']
            team_results[team]['total_gf'] += stats['goals_for']
            team_results[team]['total_ga'] += stats['goals_against']
            team_results[team]['total_gd'] += stats['goal_difference']
            team_results[team]['total_wins'] += stats['won']
            team_results[team]['total_draws'] += stats['drawn']
            team_results[team]['total_losses'] += stats['lost']
            team_results[team]['position_counts'][pos] += 1
            
            # Track special achievements
            if pos == 0:  # Champion
                team_results[team]['title_wins'] += 1
            if pos < 4:   # Top 4
                team_results[team]['top4_finishes'] += 1
            if pos >= 17:  # Bottom 3 (relegated)
                team_results[team]['relegations'] += 1
    
    # Calculate and display results
    print("\n🏆 2025-26 Premier League - Average Final Table")
    print("=" * 85)
    print(f"{'Pos':<3} {'Team':<25} {'Pts':<4} {'W':<3} {'D':<3} {'L':<3} {'GF':<4} {'GA':<4} {'GD':<5} {'Title%':<7} {'Top4%':<6} {'Rel%':<5}")
    print("-" * 85)
    
    # Calculate averages and sort
    avg_results = []
    for team, results in team_results.items():
        avg_results.append({
            'team': team,
            'avg_points': results['total_points'] / num_simulations,
            'avg_wins': results['total_wins'] / num_simulations,
            'avg_draws': results['total_draws'] / num_simulations,
            'avg_losses': results['total_losses'] / num_simulations,
            'avg_gf': results['total_gf'] / num_simulations,
            'avg_ga': results['total_ga'] / num_simulations,
            'avg_gd': results['total_gd'] / num_simulations,
            'title_percentage': (results['title_wins'] / num_simulations) * 100,
            'top4_percentage': (results['top4_finishes'] / num_simulations) * 100,
            'relegation_percentage': (results['relegations'] / num_simulations) * 100
        })
    
    avg_results.sort(key=lambda x: x['avg_points'], reverse=True)
    
    # Display final table
    for pos, team_data in enumerate(avg_results, 1):
        team = team_data['team']
        print(f"{pos:<3} {team:<25} "
              f"{team_data['avg_points']:<4.0f} "
              f"{team_data['avg_wins']:<3.0f} "
              f"{team_data['avg_draws']:<3.0f} "
              f"{team_data['avg_losses']:<3.0f} "
              f"{team_data['avg_gf']:<4.0f} "
              f"{team_data['avg_ga']:<4.0f} "
              f"{team_data['avg_gd']:<+5.0f} "
              f"{team_data['title_percentage']:<7.1f} "
              f"{team_data['top4_percentage']:<6.1f} "
              f"{team_data['relegation_percentage']:<5.1f}")
    
    # Save detailed results
    results_summary = {
        'simulation_metadata': {
            'num_simulations': num_simulations,
            'historical_matches': len(matches_df),
            'league_avg_goals': league_avg_goals,
            'home_advantage': home_advantage,
            'simulation_date': datetime.now().isoformat()
        },
        'final_table': avg_results,
        'position_probabilities': {
            team: {f'pos_{i+1}': count/num_simulations*100 
                   for i, count in enumerate(results['position_counts'])}
            for team, results in team_results.items()
        }
    }
    
    # Save to JSON for later analysis
    with open('data/processed/simulation_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to data/processed/simulation_results.json")
    
    print("\n🎯 Championship Race:")
    print("-" * 40)
    for i, team_data in enumerate(avg_results[:5]):
        print(f"{i+1}. {team_data['team']}: {team_data['title_percentage']:.1f}% chance")
    
    print("\n⬇️ Relegation Battle:")
    print("-" * 40)
    for team_data in sorted(avg_results, key=lambda x: x['relegation_percentage'], reverse=True)[:5]:
        if team_data['relegation_percentage'] > 0:
            print(f"{team_data['team']}: {team_data['relegation_percentage']:.1f}% chance")

if __name__ == "__main__":
    random.seed(42)  # For reproducible results
    np.random.seed(42)
    main()
