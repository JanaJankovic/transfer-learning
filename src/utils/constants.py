def get_countries(commodity, country=None):
    india = { 'raw': '../data/raw/India-1994-2024.csv', 'processed': '../data/processed/India-1994-2024', 'name': 'India' }
    indonesia = { 'raw': '../data/raw/Indonesia-2007-2023.csv', 'processed': '../data/processed/Indonesia-2007-2023', 'name': 'Indonesia' }
    iran = { 'raw': '../data/raw/Iran-2012-2022.csv', 'processed': '../data/processed/Iran-2012-2022', 'name': 'Iran' }
    moldova = { 'raw': '../data/raw/Moldova-2008-2024.csv', 'processed': '../data/processed/Moldova-2008-2024', 'name': 'Moldova' }
    philipines = { 'raw': '../data/raw/Philipines-2000-2024.csv', 'processed': '../data/processed/Philipines-2000-2024', 'name': 'Philipines' }
    senegal = { 'raw': '../data/raw/Senegal-2000-2023.csv', 'processed': '../data/processed/Senegal-2000-2023', 'name': 'Senegal' }
    turkey = { 'raw': '../data/raw/Turkey-2013-2022.csv', 'processed': '../data/processed/Turkey-2013-2022', 'name': 'Turkey' }
    ukraine = { 'raw': '../data/raw/Ukraine-2014-2023.csv', 'processed': '../data/processed/Ukraine-2014-2023', 'name': 'Ukraine' }

    datasets = [india, indonesia, iran, moldova, philipines, senegal, turkey, ukraine]
    for dataset in datasets:
        dataset['processed'] = dataset['processed'] + '-' + commodity + '.csv'
    
    if country is not None:
        return next((dataset for dataset in datasets if dataset['name'] == country), None)
    
    return datasets

def get_scaler_filename(country, commodity):
    return f'../scalers/{country}-{commodity}-StandardScaler.pkl'