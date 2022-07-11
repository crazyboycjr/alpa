INCPATH := -I/opt/amazon/openmpi/include -I/usr/local/cuda-11.4/include
LIBS := -L/usr/local/cuda-11.4/lib -L/opt/amazon/openmpi/lib -L/usr/local/cuda-11.4/lib64
CFLAGS := -g -O3 -Wall
LDFLAGS := -lnccl -lmpi -lcudart

send_recv: send_recv.cc
	$(CXX) $^ -o $@ $(CFLAGS) $(INCPATH) $(LIBS) $(LDFLAGS)