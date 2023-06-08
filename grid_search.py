import csv
import glob
import itertools
import logging
import os
import smtplib
import ssl
from datetime import datetime
import toml
import torch
from monai.apps import get_logger
from monai.utils import set_determinism
from tensorboardX import SummaryWriter

from eval_network import run_evaluation
from train_network import create_trainer


def run_grid_search(toml_file, output_folder=None):
    grid_search_args = toml.load(toml_file)
    params = [[(section, param, val) for val in grid_search_args[section][param]] for section in grid_search_args for param in grid_search_args[section]]
    if output_folder is None:
        output_folder = "grid_search_output/" + toml_file.replace('.toml', '').split('/')[-1].split('\\')[-1] + f"_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    os.makedirs(output_folder, exist_ok=True)

    for i, new_args in enumerate(itertools.product(*params)):
        try:
            print(f"RUNNING CONFIGURATION {i} ---------------------------------------------------------------------------------------------------------")
            print("STARTING TRAINING")

            # Load the default toml file
            args = toml.load("default.toml")
            for section, param, val in new_args:
                args[section][param] = val
                print(f"\t {param} = {val}")

            # Create the output folder
            run_output_folder = output_folder + f"/run{i}"
            os.makedirs(run_output_folder, exist_ok=True)

            # Write the resulting TOML file to the output directory
            output_toml_file = run_output_folder + "/config.toml"
            if not os.path.exists(output_toml_file):
                with open(output_toml_file, "w+") as f:
                    toml.dump(args, f)
            else:
                print("TOML file already found.")
            # Create the logger
            fmt = "[%(levelname)-5.5s][%(asctime)s] %(message)s"
            formatter = logging.Formatter(fmt)
            file_handler = logging.FileHandler(os.path.join(run_output_folder, "train_stdout.log"))
            file_handler.setFormatter(formatter)
            get_logger("train", fmt=fmt, logger_handler=file_handler)

            # Create and run the trainer
            set_determinism(args["training"]["seed"])

            last_checkpoint_file = run_output_folder + "/train_checkpoint_epoch=200.pt"
            if not os.path.exists(last_checkpoint_file):
                trainer = create_trainer(args, run_output_folder)
                trainer.run()
            else:
                print(f"Skipping training for configuration {i}")

            # Run evaluation
            print("STARTING EVALUATION")
            d = {param : val for _, param, val in new_args}

            model_path = glob.glob(os.path.join(run_output_folder, "*net_key_metric*.pt"))[0]
            metrics = run_evaluation(args, model_path, os.path.join(run_output_folder, "guidance"))

            last_gd1, last_gd2 = None, None
            for num_points, (dice, hd, bd, gd1, gd2) in enumerate(metrics):
                print(f"*** Evaluation for num_points = {num_points} finished ***")
                d[f"dice_{num_points}"] = dice
                d[f"hausdorff_{num_points}"] = hd
                d[f"boundary_dice_{num_points}"] = bd
                if num_points == 5:
                    last_gd1 = gd1
                    last_gd2 = gd2
            d["gdr_impr"] = ((last_gd1 - last_gd2) / last_gd1).mean().item()
            # Write the CSV file
            csv_file = output_folder + "/results.csv"
            print(f"WRITING RESULTS TO '{csv_file}'")
            has_header = os.path.exists(csv_file)
            with open(csv_file, 'a+', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=d.keys())
                if not has_header:
                    writer.writeheader()
                writer.writerow(d)
        except Exception as e:
            yield f"Exception occurred running configuration {i}", str(e) + "\n\n" + f"params:\n" + '\n'.join(f"{param} = {val}" for _, param, val in new_args)

    yield "Completed", "Grid search completed."


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        dest_email = "florian.ramakers@student.kuleuven.be"
        src_email = "fr.notifier@gmail.com"
        for subject, message in run_grid_search(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None):
            if dest_email is not None:
                import socket
                hostname = socket.gethostname()
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL("smtp.gmail.com", context=context) as server:
                    server.login(src_email, "anlhgtvjzduvvogr")
                    message = f"Subject: [{hostname}] {subject}\n\n" \
                              f"Message:\n" \
                              f"{message}\n\n"
                    server.sendmail(src_email, dest_email, message)
    else:
        print("Please provide a TOML file.")