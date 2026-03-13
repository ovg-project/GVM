.PHONY: build install test clean

BINARY=gvm-runc
INSTALL_PATH=/usr/local/bin

build:
	@echo "Building $(BINARY)..."
	go build -o $(BINARY) main.go

install: build
	@echo "Installing $(BINARY) to $(INSTALL_PATH)..."
	sudo cp $(BINARY) $(INSTALL_PATH)/$(BINARY)
	sudo chmod +x $(INSTALL_PATH)/$(BINARY)
	@echo "Installation complete!"
	@echo ""
	@echo "To configure Docker, add this to /etc/docker/daemon.json:"
	@echo '{'
	@echo '  "runtimes": {'
	@echo '    "gvm": {'
	@echo '      "path": "$(INSTALL_PATH)/$(BINARY)"'
	@echo '    }'
	@echo '  }'
	@echo '}'
	@echo ""
	@echo "Then restart Docker: sudo systemctl restart docker"

test:
	@echo "Running tests..."
	go test -v ./...

clean:
	@echo "Cleaning..."
	rm -f $(BINARY)

uninstall:
	@echo "Uninstalling $(BINARY)..."
	sudo rm -f $(INSTALL_PATH)/$(BINARY)
