import argparse

from config.method_config import *
import utils


if __name__ == "__main__":
    """
    Prints results of methods that were already run:

    python print_results.py --dataset dataset --method method --num-runs 5

    where

        dataset is one of: [labour, traffic, tourism, tourismlarge, wiki]
        (see config/dataset_config.py)        

        method is one of: [HierE2E, DeepVAR, DeepVARPlus, 
                            ETS_NaiveBU, ARIMA_NaiveBU,
                            ETS_MINT_shr, ETS_MINT_ols, ARIMA_MINT_shr, ARIMA_MINT_ols,
                            ETS_ERM, ARIMA_ERM,
                            PERMBU_MINT, 
                          ]
        (see config/method_config.py) 
        
        num-runs: number of runs to consider while taking the mean and std; default 5. 
        If results are available for fewer number of runs, then mean/std is calculated over only those results 
        available in `experiments/results` folder. 

    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--num-runs", type=int, required=False, default=5)

    args, _ = parser.parse_known_args()

    dataset = args.dataset
    method = args.method
    num_runs = args.num_runs

    if dataset == "wiki":
        dataset = "wiki2"
    elif dataset == "tourism":
        dataset = "tourismsmall"

    agg_metrics_ls, level_wise_agg_metrics_ls = utils.parse_results(dataset=dataset, method=method, num_runs=num_runs)

    if not agg_metrics_ls:
        print(f"No results found for {method} on {args.dataset}! "
              f"First run this method using the script: `run_experiment_with_best_hps.py`")
    else:
        utils.print_results(agg_metrics_ls=agg_metrics_ls, level_wise_agg_metrics_ls=level_wise_agg_metrics_ls)
