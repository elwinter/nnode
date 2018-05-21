#!/usr/bin/env python

# Quick script to run sets of neural networks.

import os
from subprocess import Popen, PIPE

ode = 'ode00'
ntrains = (5, 10, 15, 20)
nhids = (5, 10, 15, 20, 25, 30)
etas = (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0)

for ntrain in ntrains:
    for nhid in nhids:
        for eta in etas:
            process = Popen(
                ["./nnode1.py",
                 "--ode=%s" % ode,
                 "--ntrain=%d" % ntrain,
                 "--nhid=%d" % nhid,
                 "--eta=%f" % eta
             ], stdout=PIPE)
            (output, err) = process.communicate()
            exit_code = process.wait()
            if exit_code == 0:
                rmse = float(output)
            else:
                rmse = 'error'
            print(rmse, end=' ')
        print()
    print()
