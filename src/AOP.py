from time import time
def timing(f):
    def wrap(*args):
        time1 = time()
        ret = f(*args)
        time2 = time()
        print("Function " + str(f.__name__) + " took " + str(((time2-time1)*1000.0)) + " ms")
        if args:
            print("It had the following arguments",*args)
        else:
            print("It had no arguments")
        return ret
    return wrap
