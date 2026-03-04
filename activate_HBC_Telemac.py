import os
import subprocess
"""
run_env
=======

Python script to activate the TELEMAC and HBC environments from the activation script "activateHBCtelemac.sh"
located in the ``env-scripts`` directory.

"""
def main():
    # Directory where this script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to env-scripts/activateHBCtelemac.sh
    sh_script = os.path.join(base_dir, "env-scripts", "activateHBCtelemac.sh")

    if not os.path.exists(sh_script):
        raise FileNotFoundError(f"Script not found: {sh_script}")

    # Run the shell script
    subprocess.run(
        ["bash", sh_script],
        check=True
    )

    print("activateHBCtelemac.sh executed successfully.")


if __name__ == "__main__":
    main()