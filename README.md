# Vehicle Detection
CS 304: Computer Intelligence Project

Implementation of Sliding Window algorithm using CNN for object detection in autonomous diving cars.

* Data-set included - Vehicles and Non-Vehicles (approx 8k each).
* Sample test images included.
* Sample video included.

### Run instructions

* Download data zip files and code files.
* Extract vehicles and non-vehicles into `Data/`.

```shell
python3 ./preprocess.py
python3 ./train.py
python3 ./sw_algorithm.py
```