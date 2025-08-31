import pandas as pd
import os
from datetime import datetime

def process_season_data(file_path, season):
    """Process a single season CSV file"""
    try:
        df = pd.read_csv(file_path)
        
        # Standardize column names and extract key data
        processed_data = []
        for _, row in df.iterrows():
            if pd.notna(row.get('FTHG')) and pd.notna(row.get('FTAG')):
                # Try multiple date formats
                date_str = str(row['Date'])
                try:
                    if '/' in date_str:
                        match_date = pd.to_datetime(date_str, format='%d/%m/%Y')
                    else:
                        match_date = pd.to_datetime(date_str)
                except:
                    continue  # Skip if date can't be parsed
                
                # Calculate outcome
                if int(float(row['FTHG'])) > int(float(row['FTAG'])):
                    outcome = 'home'
                elif int(float(row['FTHG'])) < int(float(row['FTAG'])):
                    outcome = 'away'
                else:
                    outcome = 'draw'
                
                match_data = {
                    'date': match_date,
                    'home_team': str(row['HomeTeam']),
                    'away_team': str(row['AwayTeam']),
                    'home_goals': int(float(row['FTHG'])),
                    'away_goals': int(float(row['FTAG'])),
                    'outcome': outcome,
                    'season': season
                }
                processed_data.append(match_data)
        
        return pd.DataFrame(processed_data)
    except Exception as e:
        print(f'Error processing {season}: {e}')
        return pd.DataFrame()

def main():
    # Process all historical seasons
    all_seasons = []
    historical_seasons = ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25_complete']
    
    print('🔄 Processing historical seasons...')
    for season in historical_seasons:
        if season == '2024-25_complete':
            file_path = f'data/raw/epl_2024-25_complete.csv'
            season_name = '2024-25'
        else:
            file_path = f'data/raw/epl_{season}.csv'
            season_name = season
            
        if os.path.exists(file_path):
            season_df = process_season_data(file_path, season_name)
            if not season_df.empty:
                all_seasons.append(season_df)
                print(f'✅ {season_name}: {len(season_df)} matches')
            else:
                print(f'❌ {season_name}: Failed to process')
        else:
            print(f'❌ {season_name}: File not found')
    
    # Load existing current season data if it exists and has data
    try:
        existing_df = pd.read_parquet('data/processed/matches.parquet')
        if len(existing_df) > 0:
            all_seasons.append(existing_df)
            print(f'✅ Current data: {len(existing_df)} matches')
        else:
            print('⚠️ Current data: Empty file, using only historical data')
    except:
        print('⚠️ No current data file found, using only historical data')
    
    # Combine all data
    if all_seasons:
        combined_df = pd.concat(all_seasons, ignore_index=True)
        combined_df = combined_df.dropna()
        combined_df = combined_df.sort_values('date')
        
        # Save expanded dataset
        combined_df.to_parquet('data/processed/matches.parquet', index=False)
        
        print(f'\n🎯 Enhanced Training Dataset Summary:')
        print(f'Total matches: {len(combined_df)}')
        
        if len(combined_df) > 0:
            print(f'Date range: {combined_df.date.min().strftime("%Y-%m-%d")} to {combined_df.date.max().strftime("%Y-%m-%d")}')
            print(f'Seasons: {sorted(combined_df.season.unique())}')
            print(f'\nMatches per season:')
            season_counts = combined_df.groupby('season').size().sort_index()
            for season, count in season_counts.items():
                print(f'  {season}: {count} matches')
            
            print(f'\n✅ Enhanced training dataset saved with {len(combined_df)} total matches!')
            
            # Quick stats
            print(f'\n📊 Dataset Quality:')
            print(f'Average goals per match: {(combined_df.home_goals.mean() + combined_df.away_goals.mean()):.1f}')
            print(f'Home win rate: {(combined_df.home_goals > combined_df.away_goals).mean():.1%}')
            print(f'Draw rate: {(combined_df.home_goals == combined_df.away_goals).mean():.1%}')
            print(f'Away win rate: {(combined_df.home_goals < combined_df.away_goals).mean():.1%}')
        else:
            print('❌ No valid data after processing')
    else:
        print('❌ No data to combine')

if __name__ == "__main__":
    main()
