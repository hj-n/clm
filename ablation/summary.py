
import numpy as np
import helpers as hp
import metric_run as mer
import argparse

## argument setting
parser = argparse.ArgumentParser(description='Plot the summary of the ablation study', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--testtype", "-t", type=str, default="shift", help="set the type of the test ('shift' or 'card')")
parser.add_argument("--measure", "-m", type=str, default="ch", help="set the measure (ch, dunn, ii, sil, xb)")
args = parser.parse_args()
testtype_args = args.testtype 
measure_args = args.measure

sizes = np.array([5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000])
dims = dims = np.array([2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
measures = [measure_args, f"{measure_args}_range", f"{measure_args}_shift", f"{measure_args}_btw"]

keys = sizes if testtype_args == "card" else dims

print(f"plotting {testtype_args} test summary...")
hp.plot_barchart(f"{measure_args}_results_{testtype_args}/scores", measures, keys, testtype_args)
print("plotting finished!!")