# Makefile

# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -w -O3 -std=c++17 -fPIC -fopenmp

# Source files
SRCS = source/binoctree.cpp source/coarse_step.cpp \
       source/medium_step.cpp source/fine_step.cpp source/dual_contouring.cpp \
       source/bisection.cpp source/slicing.cpp

# Object directory
OBJDIR = lib

# Object files (automatically create the corresponding object file names in OBJDIR)
OBJS = $(patsubst source/%.cpp,$(OBJDIR)/%.o,$(SRCS))

# Shared library name
LIB = lib/core.so

# Default target
all: $(OBJDIR) $(LIB)

# Create the object directory
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Linking step to create the shared library
$(LIB): $(OBJS)
	$(CXX) -shared -o $(LIB) $(OBJS) -fopenmp

# Compilation step
$(OBJDIR)/%.o: source/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(OBJS) $(LIB)