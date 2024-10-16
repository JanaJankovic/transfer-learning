import utils.training as train
import utils.constants as c
import utils.transfer_learning as tl


if __name__ == "__main__":
    commodity = "Rice"
    # large_datasets = ["Bangladesh", "India", "Indonesia", "Pakistan", "Tajikistan"]
    # small_datasets = ["Afghanistan", "Lao"]

    # print("Training big models")
    # train.training_pipeline(
    #     large_datasets, commodity, c.get_large_model_results(), final=True
    # )

    # print("Training small models")
    # train.training_pipeline(small_datasets, commodity, c.get_small_model_results())

    # print("Transfer learning Afghanistan")
    # tl.transfer_learning_pipeline(
    #     "Afghanistan", large_datasets, commodity, c.get_tl_model_results()
    # )

    # print("Transfer learning Lao")
    # tl.transfer_learning_pipeline(
    #     "Lao", large_datasets, commodity, c.get_tl_model_results()
    # )

    # print("Transfer learning Market Quetta")
    # countries = ["Pakistan"]
    # train.training_pipeline(
    #     countries,
    #     commodity,
    #     c.get_market_results(),
    #     final=True,
    #     market="Quetta",
    # )

    # print("Transfer learning Market Karachi")
    # train.training_pipeline(
    #     countries,
    #     commodity,
    #     c.get_market_results(),
    #     final=False,
    #     market="Karachi",
    # )

    print("Transfer learning Market Quetta -> Karachi")
    tl.transfer_learning_market_pipeline(
        "Pakistan", commodity, "Karachi", "Quetta", c.get_market_tl_results()
    )
