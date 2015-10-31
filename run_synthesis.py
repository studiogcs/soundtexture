import sys
import wave

def open_wavefile(filename):
	print "reading " + filename 
	try:
		wavefile = wave.open(filename, 'r')
	except:
		print "ERROR: could not read file"
		sys.exit(1)

	[num_chan, samp_width, fs, num_frames, comptype, compname] = wavefile.getparams()
	wavefile.close()

	print "\t# channels: " + num_chan + "\n\tsample width: " + samp_width + "\n\tsample rate: " + fs + "\n\t# samples: " + 

if __name__ == "__main__":
	if len(sys.argv) > 1:
		filename = sys.argv[1]
		open_wavefile(filename)
	else:
		print "default"