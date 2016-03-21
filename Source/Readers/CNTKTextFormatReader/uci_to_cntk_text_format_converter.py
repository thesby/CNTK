import sys, argparse

parser = argparse.ArgumentParser(description="UCI to CNTKText format converter",
	epilog=("Quick example - converting MNIST data (see Examples/Image/MNIST):\n"
			"-in Examples/Image/MNIST/Data/Train-28x28.txt " 
			"-fs 1 -fd 784 -ls 0 -ld 1 --num_labels 10 "
			"--output_file Examples/Image/MNIST/Data/Train-28x28_cntk_text.txt"))
requiredNamed = parser.add_argument_group('required arguments');

requiredNamed.add_argument("-in", "--input_file", 
	help="input file path", required=True)
requiredNamed.add_argument("-fs", "--features_start", type=int, 
	help="features start offset", required=True)
requiredNamed.add_argument("-fd", "--features_dim", type=int, 
	help="features input size/dimension", required=True)
requiredNamed.add_argument("-ls", "--labels_start", type=int, 
	help="labels start offset", required=True)
requiredNamed.add_argument("--num_labels", type=int, 
	help="number of possible label size (labelDim parameter in reader config)", 
	required=True)

parser.add_argument("-out", "--output_file", help="output file path")
parser.add_argument("-ld", "--labels_dim", type=int, default=1, 
	help="labels input dimension (default is 1)")

args = parser.parse_args()

file_in = args.input_file
file_out = args.output_file

if not file_out:
	dot = file_in.rfind(".")
	if dot == -1:
		dot = len(file_in)
	file_out = file_in[:dot] + "_cntk_text" + file_in[dot:]

print (" Converting from UCI format\n\t '{}'\n"
	" to CNTK text format\n\t '{}'").format(file_in, file_out)

input_file = open(file_in, 'r')
output_file = open(file_out, 'w')

for line in input_file.readlines():
	values = line.split( )
	labels = values[args.labels_start:args.labels_start+args.labels_dim]
	dense_label = ['0'] * args.num_labels;
	for label in labels:		
		dense_label[int(label)] = '1';
	output_file.write("|labels " + " ".join(dense_label))
	output_file.write("\t")
	output_file.write("|features " + 
		" ".join(values[args.features_start:args.features_start+args.features_dim]))
	output_file.write("\n")

input_file.close();
output_file.close();