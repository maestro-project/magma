# MAGMA #
This is the implementation of the paper [MAGMA](https://arxiv.org/abs/2104.13997).

### Installation ###
* Install requirement
```
pip install -r requirements.txt
```
* Download cost model and build symbolic link
```
python build.py
```
### Example Usage ###
* Run MAGMA: ``sh run/runGA.sh``
* Run RL: ``sh run/runRL.sh``
* Run Blackbox: ``sh run/run_blackbox.sh``

### Contributor ###
* Sheng-Chun (Felix) Kao
* Tushar Krishna

### Citation ###
```
@inproceedings{kao2022magma,
  title={MAGMA: An Optimization Framework for Mapping Multiple DNNs on Multiple Accelerator Cores},
  author={Kao, Sheng-Chun and Krishna, Tushar},
  booktitle={2022 IEEE International Symposium on High-Performance Computer Architecture (HPCA)},
  pages={814--830},
  year={2022},
  organization={IEEE}
}

```
