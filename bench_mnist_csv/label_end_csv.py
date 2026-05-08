## The script just to put the label as the last column
import os, sys
import getopt

def parse_options():
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'i:o:s', ['in=', 'out=', 'sep='])
    except getopt.GetoptError as err:
        sys.stderr.write(str(err).capitalize())
        sys.exit()

    in_file, out_file = None, None
    sep = ','

    for opt, arg in opts:
        if opt in ('-i', '--in'):
            in_file = str(arg)
            assert os.path.isfile(in_file)
        elif opt in ('-o', '--out'):
            out_file = str(arg)
            if os.path.isfile(out_file):
                os.remove(out_file)
        elif opt in ('-s', '--sep'):
            sep = str(arg)
        else:
            assert False, 'Unhandled option: {0} {1}'.format(opt, arg)
    return in_file, out_file, sep

def parse_label_end(fp_i, fp_o, separator):
    '''
        parse the initial file and put the label into the last column
    '''
    lines = fp_i.readlines()

    for line in lines:
        infos = line.strip().split(separator)
        fp_o.write(separator.join(infos[1:]) + separator + infos[0] + '\n')


if __name__ == '__main__':
    in_file, out_file, sep = parse_options()
    
    with open(in_file, 'r') as f_in:
        with open(out_file, 'w') as f_out:\
            parse_label_end(f_in, f_out, sep)

    print('done')
    