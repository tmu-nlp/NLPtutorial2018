import pickle as pkl
import train_sr as sr

def main():
    W = pkl.load(open(sr.WEIGHT_PATH, 'rb'))

    

if __name__ == '__main__':
    main()
    # import cProfile
    # cProfile.run('main()')