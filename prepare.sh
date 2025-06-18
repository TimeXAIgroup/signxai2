#!/bin/bash
# SignXAI Setup Script
# Downloads documentation, examples, and sample data

echo "🚀 Setting up SignXAI..."

# Create directories if they don't exist
mkdir -p docs examples/data/timeseries examples/data/images examples/data/models

echo "📚 Downloading documentation from GitHub..."
if [ ! -d "docs" ] || [ -z "$(ls -A docs)" ]; then
    wget -q --show-progress -O docs.zip https://github.com/yourusername/signxai/archive/refs/heads/main.zip
    unzip -q docs.zip "*/docs/*" -d temp/
    mv temp/*/docs/* docs/
    rm -rf temp docs.zip
fi

echo "📝 Downloading examples from GitHub..."
if [ ! -d "examples" ] || [ -z "$(ls -A examples)" ]; then
    wget -q --show-progress -O examples.zip https://github.com/yourusername/signxai/archive/refs/heads/main.zip
    unzip -q examples.zip "*/examples/*" -d temp/
    mv temp/*/examples/* examples/
    rm -rf temp examples.zip
fi

echo "📊 Downloading sample ECG data..."
cd examples/data/timeseries
wget -q --show-progress https://physionet.org/files/ptb-xl/1.0.3/records500/03000/03509_hr.hea
wget -q --show-progress https://physionet.org/files/ptb-xl/1.0.3/records500/03000/03509_hr.dat
wget -q --show-progress https://physionet.org/files/ptb-xl/1.0.3/records500/12000/12131_hr.hea
wget -q --show-progress https://physionet.org/files/ptb-xl/1.0.3/records500/12000/12131_hr.dat
wget -q --show-progress https://physionet.org/files/ptb-xl/1.0.3/records500/14000/14493_hr.hea
wget -q --show-progress https://physionet.org/files/ptb-xl/1.0.3/records500/14000/14493_hr.dat
wget -q --show-progress https://physionet.org/files/ptb-xl/1.0.3/records500/02000/02906_hr.hea
wget -q --show-progress https://physionet.org/files/ptb-xl/1.0.3/records500/02000/02906_hr.dat
cd ../../..

echo "🖼️ Downloading sample images..."
cd examples/data/images
if [ ! -f "example.jpg" ]; then
    wget -q --show-progress -O example.jpg https://github.com/yourusername/signxai/raw/main/examples/data/images/example.jpg
fi
cd ../../..

echo "✅ Setup complete! Documentation in ./docs/, examples in ./examples/"
echo "📖 View docs: open docs/index.html"
echo "🧪 Run examples: python examples/comparison/compare_tf_pytorch_vgg16.py"