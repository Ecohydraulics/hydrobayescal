import os
import subprocess

"""
run_env
=======

Launches a command prompt with both the HydroBayesCal Conda environment
and TELEMAC environment activated.

Windows:
    Runs activateHBCtelemacWindows.bat

Linux:
    Runs activateHBCtelemac.sh
"""


def main():

    # Directory where this Python script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # ---------------------------------------------------------
    # Windows
    # ---------------------------------------------------------
    if os.name == "nt":

        bat_script = os.path.join(
            base_dir,
            "activateHBCtelemacWindows.bat"
        )

        if not os.path.exists(bat_script):
            raise FileNotFoundError(
                f"Activation script not found: {bat_script}"
            )

        print("Launching HydroBayesCal + TELEMAC environment...")

        # /k = execute the BAT and keep Command Prompt open
        subprocess.run(
            [
                "cmd.exe",
                "/k",
                "call",
                bat_script
            ],
            check=True
        )

    # ---------------------------------------------------------
    # Linux / Unix
    # ---------------------------------------------------------
    else:

        sh_script = os.path.join(
            base_dir,
            "activateHBCtelemac.sh"
        )

        if not os.path.exists(sh_script):
            raise FileNotFoundError(
                f"Activation script not found: {sh_script}"
            )

        subprocess.run(
            ["bash", sh_script],
            check=True
        )


if __name__ == "__main__":
    main()
