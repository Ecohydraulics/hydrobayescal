# User definitions
$TELEMAC_CONFIG_DIR = "C:\modelling\telemac\v8p5r0\configs"
$TELEMAC_CONFIG_NAME = "pysource.win.sh"
$HBCenv_DIR = "C:\USER\hydrobayescal\HBCenv"

# Get current working directory
$ACT_DIR = Get-Location

# Feedback function
function Feedback {
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  **Success**"
    } else {
        Write-Host "! ERROR: failed to source environment (check path)."
    }
}

# Function to load TELEMAC config
function Activate-Telemac {
    Write-Host "> Loading TELEMAC config..."
    Set-Location -Path $TELEMAC_CONFIG_DIR
    if (Test-Path $TELEMAC_CONFIG_NAME) {
        . .\$TELEMAC_CONFIG_NAME
    } else {
        Write-Host "! ERROR: TELEMAC config file not found."
        return -1
    }
}

# Function to activate HBCenv and its packages
function Activate-HBCenv {
    Write-Host "> Loading HBCenv..."
    Set-Location -Path $HBCenv_DIR
    if (Test-Path "bin\activate.ps1") {
        . .\bin\activate.ps1
    } else {
        Write-Host "! ERROR: HBCenv activation script not found."
        return -1
    }
}

# Run functions
Activate-HBCenv
Feedback
Activate-Telemac
Feedback

# Return to the original directory
Set-Location -Path $ACT_DIR

