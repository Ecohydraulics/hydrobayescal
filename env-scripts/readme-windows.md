# TelaPy on Windows

## Background information


The `source` command is commonly used in Unix-based systems to execute shell scripts that set up environment variables and paths. In Windows, you can achieve similar functionality by using either PowerShell or a compatible shell environment (e.g., WSL, Git Bash). To run a `.sh` file in Windows, use `.\file_name.sh`.


## Adjust the ps1 file


1. Open `activateHBCtelemacWindows.ps1` in a text editor and make sure to define the following parameters correctly according to your system settings:

    ```powershell
    $TELEMAC_CONFIG_DIR = "C:\modeling\telemac\v8p5r0\configs"
    $TELEMAC_CONFIG_NAME = "pysource.win.sh"
    $HBCenv_DIR = "C:\USER\hydrobayescal\HBCenv"
    ```

2. Save the `.ps1` file.

3. Run the `.ps1` file in PowerShell:

   ```powershell
   .\activateHBCtelemacWindows.ps1
   ```

## Testing

After setting up the environment, test if the Telemac API is working by running:

```powershell
python -c "import telapy; print(telapy.__version__)"
```

## Troubleshooting

There are a couple of issues that can be caused by the execution policy. To allow script execution, you may need to adjust your PowerShell execution policy using:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

