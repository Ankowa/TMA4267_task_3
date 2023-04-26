if test -d ".venv"
then
    echo ".venv already exist"
    source .venv/bin/activate
else
    echo "Creating .venv"
    python3 -m venv .venv
    source .venv/bin/activate
fi
pip install -r requirements.txt