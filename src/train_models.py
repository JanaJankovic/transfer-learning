import utils.training as train
import utils.constants as c
import utils.transfer_learning as tl


if __name__ == '__main__':
    commodity = 'Rice'
    large_datasets = ['Bangladesh', 'India', 'Indonesia', 'Nepal', 'Pakistan', 'Philipines', 'Senegal']
    small_datasets = ['Argentina', 'Nigeria', 'Ukraine']
    
    print('Training big models')
    train.training_pipeline(large_datasets, commodity, c.get_large_model_results(), 200)
    
    print('Training small models')
    train.training_pipeline(small_datasets, commodity, c.get_small_model_results(), 200)
    
    print('Transfer learning Argentina')
    tl.transfer_learning_pipeline('Argentina', large_datasets, commodity, c.get_tl_model_results(), 200)
    
    print('Transfer learning Nigeria')
    tl.transfer_learning_pipeline('Nigeria', large_datasets, commodity, c.get_tl_model_results(), 200)
    
    print('Transfer learning Ukraine')
    tl.transfer_learning_pipeline('Ukraine', large_datasets, commodity, c.get_tl_model_results(), 200)
    
    