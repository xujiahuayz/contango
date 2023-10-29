# contango

## Setup

```
git clone https://github.com/xujiahuayz/contango.git
cd contango
```

### Give execute permission to your script and then run `setup_repo.sh`

```
chmod +x setup_repo.sh
./setup_repo.sh
. venv/bin/activate
```

or follow the step-by-step instructions below between the two horizontal rules:

---

#### Create a python virtual environment

- MacOS / Linux

```bash
python3 -m venv venv
```

- Windows

```bash
python -m venv venv
```

#### Activate the virtual environment

- MacOS / Linux

```bash
. venv/bin/activate
```

- Windows (in Command Prompt, NOT Powershell)

```bash
venv\Scripts\activate.bat
```

#### Install toml

```
pip install toml
```

#### Install the project in editable mode

```bash
pip install -e ".[dev]"
```

## Set up the environmental variables

put your `COINGLASS_SECRET` in `.env`:

```
COINGLASS_SECRET="abc123"
KAIKO_API_KEY="abc123"
```

```
export $(cat .env | xargs)
```

## Running the simulation

<!-- TODO -->

## Git Large File Storage (Git LFS)

All files in [`data/`](data/) are stored with `lfs`:

```
git lfs track data/**/*
```

## Test the code

```zsh
pytest
```
