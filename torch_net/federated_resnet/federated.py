"""
Federated Learning Training Script for Comparing Real and Complex ResNets

This script uses Flower to simulate a federated learning environment for the
ResNet comparison experiment.

This version is adapted to use the classic Flower simulation API to ensure
compatibility with the user's environment. It uses rich progress bars for
monitoring training.

Usage:
    # To run a default federated experiment with 10 clients for 5 rounds:
    python federated.py --num_clients 10 --num_rounds 5

    # To run only the 'WS' complex model with crelu:
    python federated.py --num_clients 10 --architectures WS --activations crelu --learn_imag_mode true_only
"""
import flwr as fl
import torch
import argparse
from itertools import product
from datetime import datetime
import os
import csv
from typing import Dict, Any, List, Tuple

# Rich imports for better terminal output
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.panel import Panel

from federated_utils import get_model, load_partitioned_data, train, test, get_weights, set_weights

# --- FLOWER CLIENT DEFINITION ---

class FlowerClient(fl.client.NumPyClient):
    """
    A Flower client that handles training and evaluation of a ResNet model.
    It is instantiated with a client ID, model, data loaders, and a device.
    """
    def __init__(self, cid, model, trainloader, valloader, device):
        self.cid = cid
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device

    def get_parameters(self, config):
        """Return the current local model weights."""
        return get_weights(self.model)

    def fit(self, parameters, config):
        """
        Receive model parameters from the server, train the model locally,
        and return the updated model parameters.
        """
        set_weights(self.model, parameters)
        epochs = config["local_epochs"]
        learning_rate = config["learning_rate"]
        train(self.model, self.trainloader, epochs=epochs, device=self.device, learning_rate=learning_rate, cid=self.cid)
        return get_weights(self.model), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        """
        Receive model parameters from the server, evaluate the model on the
        local validation set, and return the results.
        """
        set_weights(self.model, parameters)
        loss, metrics = test(self.model, self.valloader, device=self.device)
        return float(loss), len(self.valloader.dataset), {"accuracy": float(metrics["accuracy"])}

# --- RICH-ENABLED FEDAVG STRATEGY ---

class RichFedAvg(fl.server.strategy.FedAvg):
    """
    A custom FedAvg strategy that integrates with Rich progress bars.
    """
    def __init__(self, *, progress: Progress, client_task_id: int, **kwargs):
        self.progress = progress
        self.client_task_id = client_task_id
        super().__init__(**kwargs)

    def configure_fit(
        self, server_round: int, parameters: fl.common.NDArrays, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        """Configure the next round of training and update the client progress bar."""
        clients_and_configs = super().configure_fit(server_round, parameters, client_manager)
        self.progress.update(self.client_task_id, total=len(clients_and_configs), description=f"[cyan]Fitting round {server_round}", completed=0)
        return clients_and_configs

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]], failures: List[Any]
    ) -> Tuple[fl.common.NDArrays | None, Dict[str, Any]]:
        """Aggregate fit results and advance the client progress bar."""
        self.progress.update(self.client_task_id, advance=len(results))
        return super().aggregate_fit(server_round, results, failures)


# --- SIMULATION SETUP ---

def get_evaluate_fn(test_loader, device, config, progress: Progress, rounds_task_id: int):
    """Return an evaluation function for server-side evaluation."""
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config_dict: dict):
        """Centralized evaluation function."""
        model = get_model(config)
        set_weights(model, parameters)
        model.to(device)
        
        loss, metrics = test(model, test_loader, device)
        
        progress.update(rounds_task_id, advance=1, description=f"[green]Finished round {server_round}")
        
        console.print(f"Round {server_round} | Server-side Accuracy: {metrics['accuracy']:.4f} | Loss: {loss:.4f}")
        return loss, metrics
    return evaluate

def save_results_to_csv(results: Dict[str, Any], filename="federated_results.csv"):
    """Append a dictionary of results to a CSV file."""
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = list(results.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)

# --- MAIN DRIVER ---
console = Console()

def main(args):
    """Main function to set up and run the federated learning simulation."""
    console.print(Panel("[bold green]Step 1: Initializing Setup[/bold green]"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    console.print(f"Federated training starting on device: [yellow]{device}[/yellow]")

    console.print(Panel("[bold green]Step 2: Loading and Partitioning Data[/bold green]"))
    trainloaders, valloaders, testloader = load_partitioned_data(args.num_clients, args.batch_size)
    console.print(f"Data partitioned for [cyan]{args.num_clients}[/cyan] clients.")

    console.print(Panel("[bold green]Step 3: Defining Experiment Configurations[/bold green]"))
    arch_types = args.architectures
    complex_activations = args.activations
    learn_imag_opts = {'true_only': [True], 'false_only': [False], 'both': [True, False]}[args.learn_imag_mode]
        
    experiment_configs = []
    for arch, act, learn in product(arch_types, complex_activations, learn_imag_opts):
        experiment_configs.append({'name': f"Complex_ResNet_{arch}_{act}_{'LearnImag' if learn else 'ZeroImag'}", 'model_type': 'Complex', 'arch': arch, 'activation': act, 'learn_imag': learn})
    for arch in arch_types:
        experiment_configs.append({'name': f"Real_ResNet_{arch}", 'model_type': 'Real', 'arch': arch, 'activation': 'relu', 'learn_imag': 'N/A'})
    console.print(f"Generated [cyan]{len(experiment_configs)}[/cyan] experiment configurations to run.")

    console.print(Panel("[bold green]Step 4: Starting Federated Learning Simulation Loop[/bold green]"))
    
    progress_columns = [TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeRemainingColumn(), TimeElapsedColumn()]
    
    with Progress(*progress_columns, console=console) as progress:
        exp_task_id = progress.add_task("[magenta]Overall Progress", total=len(experiment_configs))

        for config in experiment_configs:
            start_time = datetime.now()
            console.print(Panel(f"[bold]STARTING EXPERIMENT: {config['name']}[/bold]", style="cyan", border_style="cyan"))
            
            rounds_task_id = progress.add_task("[green]Federated Rounds", total=args.num_rounds)
            client_task_id = progress.add_task("[cyan]Fitting clients", total=args.min_fit_clients)

            # Define the client function, capturing variables from the loop's scope.
            # This is the pattern expected by older versions of Flower's start_simulation.
            def client_fn(cid: str) -> fl.client.Client:
                model = get_model(config).to(device)
                trainloader = trainloaders[int(cid)]
                valloader = valloaders[int(cid)]
                # Use .to_client() to avoid deprecation warnings
                return FlowerClient(cid=cid, model=model, trainloader=trainloader, valloader=valloader, device=device).to_client()

            # Define the evaluation function and strategy directly inside the loop
            evaluate_fn = get_evaluate_fn(testloader, device, config, progress, rounds_task_id)
            strategy = RichFedAvg(
                progress=progress,
                client_task_id=client_task_id,
                fraction_fit=args.fraction_fit,
                min_fit_clients=args.min_fit_clients,
                min_available_clients=args.num_clients,
                evaluate_fn=evaluate_fn,
                on_fit_config_fn=lambda server_round: {"local_epochs": args.local_epochs, "learning_rate": args.learning_rate},
            )

            console.print(f"\n--> Starting simulation for '{config['name']}'...")
            
            # Use the classic start_simulation API
            history = fl.simulation.start_simulation(
                client_fn=client_fn,
                num_clients=args.num_clients,
                config=fl.server.ServerConfig(num_rounds=args.num_rounds),
                strategy=strategy,
                client_resources={"num_gpus": 1} if device.type == "cuda" else None,
            )
            
            progress.remove_task(rounds_task_id)
            progress.remove_task(client_task_id)
            console.print(f"--- Simulation for '[bold]{config['name']}[/bold]' finished. ---", style="green")

            console.print(Panel("[bold green]Step 5: Processing and Saving Results[/bold green]"))
            final_metrics = history.metrics_centralized
            final_metrics_processed = {key: val[-1][1] for key, val in final_metrics.items() if val}
            
            end_time = datetime.now()
            final_save_data = {'status': 'Completed', **config, 'num_clients': args.num_clients, 'num_rounds': args.num_rounds, 'local_epochs': args.local_epochs, **final_metrics_processed, 'training_time_seconds': (end_time - start_time).total_seconds()}
            save_results_to_csv(final_save_data)
            console.print(f"Results for '[bold]{config['name']}[/bold]' saved. Final Accuracy: [bold yellow]{final_metrics_processed.get('accuracy', -1):.4f}[/bold yellow]")
            
            progress.update(exp_task_id, advance=1)

    console.print(Panel("[bold]All Experiments Finished[/bold]", style="bold green"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Federated Learning for ResNet Comparison")
    
    parser.add_argument("--num_clients", type=int, default=10, help="Total number of clients to simulate.")
    parser.add_argument("--num_rounds", type=int, default=5, help="Number of federated learning rounds.")
    parser.add_argument("--local_epochs", type=int, default=5, help="Number of local epochs for each client.")
    parser.add_argument("--fraction_fit", type=float, default=1.0, help="Fraction of clients to use for training.")
    parser.add_argument("--min_fit_clients", type=int, default=2, help="Minimum number of clients to train on.")
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for client training.')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning rate for client training.')
    parser.add_argument('--architectures', '--arch', nargs='+', default=['WS', 'DN', 'IB'], choices=['WS', 'DN', 'IB'], help="Architectures to train.")
    parser.add_argument('--activations', '--act', nargs='+', default=['crelu', 'zrelu', 'modrelu', 'complex_cardioid'], choices=['crelu', 'zrelu', 'modrelu', 'complex_cardioid'], help="Complex activation functions to test.")
    parser.add_argument('--learn_imag_mode', default='both', choices=['true_only', 'false_only', 'both'], help="Controls which 'learn_imaginary_component' settings to use.")
    
    args = parser.parse_args()
    main(args)
