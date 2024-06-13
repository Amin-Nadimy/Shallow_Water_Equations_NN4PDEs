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


https://github.com/Amin-Nadimy/Shallow_Water_Equations_NN4PDEs/assets/71018515/f44fed0b-9b45-499a-8383-9e65b6193202



Usage
Detailed instructions on how to use the code and run simulations can be found in the docs directory.

## License
This project is licensed under the MIT License. See the LICENSE.md file for details.

```ruby
git clone https://github.com/Amin-Nadimy/Shallow_Water_Equations_NN4PDEs.git
```
```ruby
cd Shallow_Water_Equations_NN4PDEs
```

## Contact and references
For more information please contact me via:
- Email: amin.nadimy19@imperial.ac.uk


