def get_countries(commodity):
    india = { 'raw': '../data/raw/India-1994-2024.csv', 'processed': '../data/processed/India-1994-2024' }
    indonesia = { 'raw': '../data/raw/Indonesia-2007-2023.csv', 'processed': '../data/processed/Indonesia-2007-2023' }
    iran = { 'raw': '../data/raw/Iran-2012-2022.csv', 'processed': '../data/processed/Iran-2012-2022' }
    moldova = { 'raw': '../data/raw/Moldova-2008-2024.csv', 'processed': '../data/processed/Moldova-2008-2024' }
    philipines = { 'raw': '../data/raw/Philipines-2000-2024.csv', 'processed': '../data/processed/Philipines-2000-2024' }
    senegal = { 'raw': '../data/raw/Senegal-2000-2023.csv', 'processed': '../data/processed/Senegal-2000-2023' }
    turkey = { 'raw': '../data/raw/Turkey-2013-2022.csv', 'processed': '../data/processed/Turkey-2013-2022' }
    ukraine = { 'raw': '../data/raw/Ukraine-2014-2023.csv', 'processed': '../data/processed/Ukraine-2014-2023' }

    datasets = [india, indonesia, iran, moldova, philipines, senegal, turkey, ukraine]
    for dataset in datasets:
        dataset['processed'] = dataset['processed'] + '-' + commodity + '.csv'
    return datasets