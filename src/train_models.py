import utils.training as train
import utils.constants as c
import utils.transfer_learning as tl


if __name__ == "__main__":
    commodity = "Rice"
    large_datasets = ["Bangladesh", "India", "Indonesia", "Pakistan", "Tajikistan"]
    small_datasets = ["Afghanistan", "Kyrgyzstan", "SriLanka"]

    # print("Training big models")
    # train.training_pipeline(
    #     large_datasets, commodity, c.get_large_model_results(), final=True
    # )

    print("Training small models")
    train.training_pipeline(small_datasets, commodity, c.get_small_model_results())

    print("Transfer learning Afghanistan")
    tl.transfer_learning_pipeline(
        "Afghanistan", large_datasets, commodity, c.get_tl_model_results()
    )

    print("Transfer learning Kyrgyzstan")
    tl.transfer_learning_pipeline(
        "Kyrgyzstan", large_datasets, commodity, c.get_tl_model_results()
    )

    print("Transfer learning Sri Lanka")
    tl.transfer_learning_pipeline(
        "SriLanka", large_datasets, commodity, c.get_tl_model_results()
    )
