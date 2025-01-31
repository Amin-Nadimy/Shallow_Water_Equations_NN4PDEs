# Shallow Water Equations Discretisation and Solution using Neural Networks

This repository presents a novel approach to the discretisation and solution of the Shallow Water Equations using finite difference (FD), finite volume (FV), and finite element methods (FE). Our method reformulates the discretised system of equations as a discrete convolution, analogous to a convolutional layer in a neural network, and solves it using Jacobi iteration.

## Key Features:
- **Platform-Agnostic Code**: Runs seamlessly on CPUs, GPUs, and AI-optimized processors.
- **Neural Network Integration**: Easily integrates with trained neural networks for sub-grid-scale models, surrogate models, or physics-informed approaches.
- **Accelerated Development**: Leverages machine-learning libraries to speed up model development.

## Applications:
- **Idealised Problems**: Validated against analytical solutions.
- **Laboratory Experiments**: Tested with controlled experimental data.
- **Real-World Test Case**: Applied to the 2005 Carlisle flood, demonstrating significant potential speed-ups with AI processors.

## Getting Started
To get started with the code, clone this repository using the code at the bottom of this page or click on each file to download them separately or click on the links below:
- [Carlisle Bathymetry](https://github.com/Amin-Nadimy/Shallow_Water_Equations_NN4PDEs/blob/main/carlisle-5m.dem.raw)
- [Inlet Points](https://github.com/Amin-Nadimy/Shallow_Water_Equations_NN4PDEs/blob/main/carlisle.bci)
- [Flowrates](https://github.com/Amin-Nadimy/Shallow_Water_Equations_NN4PDEs/blob/main/flowrates.csv)

Create a folder on your computer using the terminal and navigate to the folder:
```ruby
mkdir Shallow_Water_Equations_NN4PDEs
```
```ruby
cd Shallow_Water_Equations_NN4PDEs
```
Clone the repository in your folder:
```ruby
git clone https://github.com/Amin-Nadimy/Shallow_Water_Equations_NN4PDEs.git
```
---
### Carlisle 2005 Flood Event Modeled with NN4PDEs
```geojson
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "id": 1,
      "properties": {
        "ID": 0
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
              [-2.886863, 54.910878],
              [-2.959449, 54.910878],
              [-2.959449, 54.883988],
              [-2.886863, 54.883988],
              [-2.886863, 54.910878]
          ]
        ]
      }
    }
  ]
}
```

<!-- <img src="https://github.com/Amin-Nadimy/Shallow_Water_Equations_-SWE-/blob/main/SWE_2.gif" width="512" /> -->

---
## Repository Structure
```plaintext
. Shallow_Water_Equations_NN4PDEs
├── Demos/                                 # Folder for animation files
│   ├── 68-hours_simulation.mp4            # mp4 format of the 68 h simulation results
│   └── SWE_2_gif.gif                      # gif format of the 68 h simulation results
├── Documents/                             # Model inputs
│   ├── Carlisle.bci                       # A raster file of the domain
│   └── point_readings_paper.csv           # Coordinates of the reading points in the domain
├── Source_code/                           # Python codes for Linea, Quadratic and parallel simulations
│   ├── Linear/                     
│           ├── AI4SWE_Linear.py           # Main code
│           └── SWE_2D_diff_Linear.py      # External library
│   ├── Parallel/                     
│           └── SWE_parallel.py            # Main code
│   └── Quadratic/                     
│           ├── AI4SWE_quadratic.py        # Main code
│           └── SWE_2D_diff_Q.py           # External library
```
---

https://github.com/Amin-Nadimy/Shallow_Water_Equations_NN4PDEs/assets/71018515/f44fed0b-9b45-499a-8383-9e65b6193202

## Installation

### Prerequisites

Before proceeding, ensure that you have the following:

- **Python 3.10**: It is recommended to use Python 3.10 for compatibility with the required packages and libraries.

- **Conda (Preferred)**: Although not essential, it is recommended to use [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) for managing your Python environment and dependencies. Conda simplifies package management and helps avoid conflicts.

- **GPU with CUDA support**: A GPU that supports CUDA and has at least 20GB of VRAM. Ensure that your CUDA drivers are correctly installed and configured.

### Environment Setup

To set up the environment, run:

```bash
conda env create -f environment.yml
```

Alternatively, you can install the required packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

Usage
Detailed instructions on how to use the code and run simulations can be found in the docs directory.

## License
This project is licensed under the MIT License. See the LICENSE.md file for details.



## Contact and references
For more information please contact me via:
- Email: amin.nadimy19@imperial.ac.uk


