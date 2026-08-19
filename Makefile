.PHONY: all clean test

all:
	@mkdir -p build
	@cd build && cmake ..
	@cd build && $(MAKE)

test:
	python3 -m pytest tests/ -q

clean:
	rm -rf build nyangrad/_cpu_backend*.so nyangrad/_cuda_backend*.so
