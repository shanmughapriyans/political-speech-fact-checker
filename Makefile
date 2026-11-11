# Makefile  —  Developer convenience targets

.PHONY: dev demo test eval clean install-m7

# Start all Python microservices locally (requires 6 terminals or use tmux)
dev:
	@echo "Starting all modules..."
	@echo "Open 6 terminals and run:"
	@echo "  cd module1_speech_asr && python server.py --input file --file test_tone.wav"
	@echo "  cd module2_claim_detection && python pipeline.py"
	@echo "  cd module3_query_formulation && python server.py"
	@echo "  cd module4_evidence_retrieval && python server.py"
	@echo "  cd module5_rag_verification && python server.py"
	@echo "  cd module6_verdict_resolution && python server.py"
	@echo "  cd module7_dashboard && npm run dev"

# Offline demo — no live audio needed
demo:
	cd module2_claim_detection && python dump_segments.py --source synthetic --out data/sample_recording.jsonl
	cd module2_claim_detection && python replay.py --speed 0 --evaluate

# Run all self-tests
test:
	cd module2_claim_detection && python sentence_segmenter.py
	cd module2_claim_detection && python deduper.py
	cd module3_query_formulation && python decomposer.py
	cd module4_evidence_retrieval && python retriever.py
	cd module5_rag_verification && python nli_scorer.py
	cd module5_rag_verification && python rag_verifier.py
	cd module6_verdict_resolution && python resolver.py
	@echo "All self-tests passed."

# Full end-to-end evaluation
eval:
	cd module2_claim_detection && python dump_segments.py --source synthetic --out data/sample_recording.jsonl
	python module8_infrastructure/eval/end_to_end_eval.py

# Install Next.js dependencies
install-m7:
	cd module7_dashboard && npm install

# Start the dashboard (after all Python services are running)
dashboard:
	cd module7_dashboard && npm run dev

# Docker Compose full system
up:
	docker compose -f module8_infrastructure/docker-compose.yml --profile full up

down:
	docker compose -f module8_infrastructure/docker-compose.yml down

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true

