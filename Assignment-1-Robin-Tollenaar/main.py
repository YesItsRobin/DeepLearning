from mynetwork import MyNetwork

p1=[[[0,1,2,3],0],[[1,1,2,3],2],[[2,1,2,3],4],[[3,1,2,3],6],[[4,1,2,3],8]]

def main():
    net=MyNetwork([1],[0],100,1,4,1)
    net.train(p1)

if __name__ == "__main__":
    main()