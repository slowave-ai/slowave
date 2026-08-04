# Slowave service management
#   make           — restart dashboard (default)
#   make status    — show what's running
#   make restart   — restart both dashboard + MCP daemon
#   make mcp       — restart MCP daemon only

.PHONY: status restart mcp _dash

# Default: restart dashboard only
_dash:
	@echo "--- Restarting dashboard ---"
	@for port in 8765 8420; do \
		pid=$(lsof -i :$port -t 2>/dev/null | head -1); \
		if [ -n "$pid" ]; then \
			echo "Killing dashboard on :$port (PID $pid)..."; \
			kill $pid 2>/dev/null || true; \
		fi; \
	done; \
	sleep 1
	@nohup slowave dashboard --no-open > /tmp/slowave-dashboard.log 2>&1 &
	@sleep 2
	@echo "Dashboard: http://127.0.0.1:8765"

status:
	@echo "=== Slowave Status ==="
	@echo "Git branch: $(git branch --show-current 2>/dev/null || echo unknown)"
	@echo "MCP daemon PID file: $(cat ~/.slowave/daemon.pid 2>/dev/null || echo none)"
	@echo -n "Dashboard (8765): "; lsof -i :8765 -t 2>/dev/null | head -1 || echo "not listening"
	@echo -n "Dashboard (8420): "; lsof -i :8420 -t 2>/dev/null | head -1 || echo "not listening"
	@echo -n "MCP (8766): "; lsof -i :8766 -t 2>/dev/null | head -1 || echo "not listening"
	@echo -n "Worker: "; pgrep -f 'slowave worker' 2>/dev/null || echo "not running"
	@echo ""

restart: mcp _dash
	@echo "=== Done ==="

mcp:
	@echo "--- Restarting MCP daemon ---"
	@if [ -f ~/.slowave/daemon.pid ]; then \
		PID=$(cat ~/.slowave/daemon.pid); \
		if kill -0 $PID 2>/dev/null; then \
			echo "Stopping MCP daemon (PID $PID)..."; \
			kill $PID 2>/dev/null; \
			sleep 1; \
			if kill -0 $PID 2>/dev/null; then \
				echo "Force killing..."; \
				kill -9 $PID 2>/dev/null; \
			fi; \
			rm -f ~/.slowave/daemon.pid; \
		else \
			echo "Stale PID file ($PID not alive), removing..."; \
			rm -f ~/.slowave/daemon.pid; \
		fi; \
	fi
	@echo "Starting MCP daemon..."
	@nohup slowave serve start --foreground > /tmp/slowave-mcp.log 2>&1 &
	@sleep 2
	@echo "MCP daemon started (log: /tmp/slowave-mcp.log)"