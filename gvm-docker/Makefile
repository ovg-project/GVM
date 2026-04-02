.PHONY: build build-daemon build-runc install test clean uninstall

INSTALL_PATH=/usr/local/bin

build: build-daemon build-runc

build-daemon:
	@echo "Building gvm-daemon..."
	go build -o gvm-daemon ./cmd/gvm-daemon/

build-runc:
	@echo "Building gvm-runc..."
	go build -o gvm-runc ./cmd/gvm-runc/

install: build
	@echo "Installing binaries to $(INSTALL_PATH)..."
	sudo cp gvm-daemon $(INSTALL_PATH)/gvm-daemon
	sudo cp gvm-runc $(INSTALL_PATH)/gvm-runc
	sudo chmod +x $(INSTALL_PATH)/gvm-daemon $(INSTALL_PATH)/gvm-runc
	@echo ""
	@echo "Installation complete!"
	@echo ""
	@echo "=== GVM Docker Daemon ==="
	@echo "Start the daemon:"
	@echo "  sudo gvm-daemon"
	@echo ""
	@echo "=== GVM Runc Wrapper (alternative) ==="
	@echo "Add this to /etc/docker/daemon.json:"
	@echo '{'
	@echo '  "runtimes": {'
	@echo '    "gvm": {'
	@echo '      "path": "$(INSTALL_PATH)/gvm-runc"'
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
	rm -f gvm-daemon gvm-runc

uninstall:
	@echo "Uninstalling..."
	sudo rm -f $(INSTALL_PATH)/gvm-daemon $(INSTALL_PATH)/gvm-runc
