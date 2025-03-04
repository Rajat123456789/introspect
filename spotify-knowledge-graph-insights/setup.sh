#!/bin/bash

# Spotify Knowledge Graph Setup Script

# Set up colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored status messages
function echo_status() {
    echo -e "${GREEN}[*] $1${NC}"
}

function echo_warning() {
    echo -e "${YELLOW}[!] $1${NC}"
}

function echo_error() {
    echo -e "${RED}[!] $1${NC}"
}

# Check if Neo4j is installed
echo_status "Checking if Neo4j is installed..."
if ! command -v neo4j &> /dev/null; then
    echo_error "Neo4j is not installed or not in your PATH"
    echo_warning "Please install Neo4j from https://neo4j.com/download/"
    exit 1
fi

# Check if Python is installed
echo_status "Checking if Python is installed..."
if ! command -v python3 &> /dev/null; then
    echo_error "Python 3 is not installed or not in your PATH"
    exit 1
fi

# Check if pip is installed
echo_status "Checking if pip is installed..."
if ! command -v pip3 &> /dev/null; then
    echo_error "pip3 is not installed or not in your PATH"
    exit 1
fi

# Install required Python packages
echo_status "Installing required Python packages..."
pip3 install -r requirements.txt

# Ask for Neo4j credentials
echo_status "Please enter your Neo4j credentials:"
read -p "Neo4j URI [bolt://localhost:7687]: " uri
uri=${uri:-bolt://localhost:7687}

read -p "Neo4j Username [neo4j]: " username
username=${username:-neo4j}

read -sp "Neo4j Password [password]: " password
password=${password:-password}
echo

# Update scripts with credentials
echo_status "Updating scripts with your Neo4j credentials..."

# Update spotify_neo4j_graph.py
sed -i.bak "s|uri = \"bolt://localhost:7687\"|uri = \"$uri\"|g" spotify_neo4j_graph.py
sed -i.bak "s|username = \"neo4j\"|username = \"$username\"|g" spotify_neo4j_graph.py
sed -i.bak "s|password = \"password\"|password = \"$password\"|g" spotify_neo4j_graph.py

# Update query_insights.py
sed -i.bak "s|uri = \"bolt://localhost:7687\"|uri = \"$uri\"|g" query_insights.py
sed -i.bak "s|username = \"neo4j\"|username = \"$username\"|g" query_insights.py
sed -i.bak "s|password = \"password\"|password = \"$password\"|g" query_insights.py

# Remove backup files
rm *.bak 2>/dev/null

# Make run_cypher.py and this script executable
chmod +x run_cypher.py
chmod +x setup.sh

echo_status "Setup complete!"
echo_status "You can now run the following commands:"
echo
echo "  # Create the knowledge graph:"
echo "  python3 spotify_neo4j_graph.py"
echo
echo "  # Generate insights and visualizations:"
echo "  python3 query_insights.py"
echo
echo "  # Run custom Cypher queries:"
echo "  ./run_cypher.py \"MATCH (n) RETURN labels(n) as label, count(n) as count GROUP BY label\""
echo

# Ask if user wants to create the knowledge graph now
read -p "Do you want to create the knowledge graph now? (y/n): " create_now
if [[ $create_now == "y" || $create_now == "Y" ]]; then
    echo_status "Creating knowledge graph..."
    python3 spotify_neo4j_graph.py
    
    echo
    
    # Ask if user wants to generate insights now
    read -p "Do you want to generate insights and visualizations now? (y/n): " insights_now
    if [[ $insights_now == "y" || $insights_now == "Y" ]]; then
        echo_status "Generating insights and visualizations..."
        python3 query_insights.py
    fi
fi

echo_status "Done!" 