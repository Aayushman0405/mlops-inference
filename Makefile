.PHONY: help setup deploy-mlflow deploy-blue deploy-green switch test monitor logs-blue logs-green scale-blue scale-green clean

help:
	@echo "Available commands:"
	@echo "  setup         - Create namespaces and storage"
	@echo "  deploy-mlflow - Deploy MLflow tracking server"
	@echo "  deploy-blue   - Deploy blue environment"
	@echo "  deploy-green  - Deploy green environment"
	@echo "  switch        - Switch between blue/green"
	@echo "  test          - Run tests against active environment"
	@echo "  monitor       - Show monitoring info"
	@echo "  logs-blue     - Follow blue deployment logs"
	@echo "  logs-green    - Follow green deployment logs"
	@echo "  scale-blue    - Scale blue deployment (use replicas=X)"
	@echo "  scale-green   - Scale green deployment (use replicas=X)"
	@echo "  clean         - Clean up deployments"

setup:
	@echo "Setting up namespaces and storage..."
	kubectl apply -f namespaces/
	kubectl apply -f storage/

deploy-mlflow:
	@echo "Deploying MLflow tracking server..."
	kubectl apply -f mlflow/
	@echo "Waiting for MySQL to be ready..."
	kubectl wait --for=condition=available --timeout=300s deployment/mysql -n mlops-blue || true
	@echo "Waiting for MLflow to be ready..."
	kubectl wait --for=condition=available --timeout=300s deployment/mlflow-server -n mlops-blue || true
	@echo "MLflow deployed. Get credentials with: kubectl get secret mlflow-artifacts -n mlops-blue -o yaml"

deploy-blue:
	@echo "Deploying blue environment..."
	kubectl apply -f inference/01-configmap.yaml
	kubectl apply -f inference/02-secrets.yaml
	kubectl apply -f inference/03-deployment-blue.yaml
	kubectl apply -f inference/05-service.yaml
	kubectl apply -f inference/06-ingress.yaml
	@echo "Waiting for blue deployment..."
	kubectl wait --for=condition=available --timeout=300s deployment/inference-blue -n mlops-blue || true
	@echo "Blue deployment complete"

deploy-green:
	@echo "Deploying green environment..."
	kubectl apply -f inference/04-deployment-green.yaml
	@echo "Waiting for green deployment..."
	kubectl wait --for=condition=available --timeout=300s deployment/inference-green -n mlops-green || true
	@echo "Green deployment complete"

switch:
	@echo "Switching environments..."
	./scripts/switch-blue-green.sh

test:
	@echo "Testing inference endpoint..."
	./scripts/test-endpoint.sh

monitor:
	@echo "Monitoring pods (Ctrl+C to stop)..."
	kubectl get pods -n mlops-blue -w

logs-blue:
	kubectl logs -f -l app=inference,version=blue -n mlops-blue

logs-green:
	kubectl logs -f -l app=inference,version=green -n mlops-green

scale-blue:
	kubectl scale deployment inference-blue -n mlops-blue --replicas=$(replicas)

scale-green:
	kubectl scale deployment inference-green -n mlops-green --replicas=$(replicas)

clean:
	@echo "Cleaning up deployments..."
	-kubectl delete ns mlops-blue
	-kubectl delete ns mlops-green
	@echo "Done"
