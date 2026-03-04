#!/bin/bash

# Blue-Green Deployment Switch Script
set -e

NAMESPACE_BLUE="mlops-blue"
NAMESPACE_GREEN="mlops-green"
SERVICE_NAME="inference-service"
INGRESS_NAME="inference-ingress"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Blue-Green Deployment Switch ===${NC}"

# Get current active environment
CURRENT_SELECTOR=$(kubectl get svc ${SERVICE_NAME} -n ${NAMESPACE_BLUE} -o jsonpath='{.spec.selector.version}')
echo -e "${BLUE}Current active environment: ${CURRENT_SELECTOR}${NC}"

if [ "$CURRENT_SELECTOR" == "blue" ]; then
    NEW_ENV="green"
    NEW_NAMESPACE=${NAMESPACE_GREEN}
else
    NEW_ENV="blue"
    NEW_NAMESPACE=${NAMESPACE_BLUE}
fi

echo -e "${GREEN}Switching to: ${NEW_ENV}${NC}"

# Check if new environment is ready
echo -e "${YELLOW}Checking readiness of ${NEW_ENV} deployment...${NC}"
READY_REPLICAS=$(kubectl get deployment inference-${NEW_ENV} -n ${NEW_NAMESPACE} -o jsonpath='{.status.readyReplicas}')
DESIRED_REPLICAS=$(kubectl get deployment inference-${NEW_ENV} -n ${NEW_NAMESPACE} -o jsonpath='{.spec.replicas}')

if [ "$READY_REPLICAS" != "$DESIRED_REPLICAS" ]; then
    echo -e "${RED}Error: ${NEW_ENV} deployment not ready (${READY_REPLICAS}/${DESIRED_REPLICAS} pods)${NC}"
    exit 1
fi

echo -e "${GREEN}${NEW_ENV} deployment is ready${NC}"

# Run tests against new environment
echo -e "${YELLOW}Running smoke tests against new environment...${NC}"

# Port-forward to test
kubectl port-forward service/inference-service-${NEW_ENV} -n ${NEW_NAMESPACE} 8080:80 &
PF_PID=$!
sleep 3

# Test endpoint
API_KEY=$(kubectl get secret inference-secrets -n ${NAMESPACE_BLUE} -o jsonpath='{.data.API_KEY}' | base64 -d)

TEST_RESULT=$(curl -s -X POST http://localhost:8080/predict \
    -H "x-api-key: ${API_KEY}" \
    -H "Content-Type: application/json" \
    -d '{
        "vendor_id": 1,
        "passenger_count": 1,
        "pickup_longitude": -73.98,
        "pickup_latitude": 40.75,
        "dropoff_longitude": -73.97,
        "dropoff_latitude": 40.76,
        "trip_distance": 1.5,
        "pickup_hour": 12,
        "pickup_day": 2,
        "store_and_fwd_flag_Y": 0
    }')

kill $PF_PID

if [[ $TEST_RESULT == *"prediction"* ]]; then
    echo -e "${GREEN}Smoke tests passed!${NC}"
else
    echo -e "${RED}Smoke tests failed! Response: ${TEST_RESULT}${NC}"
    exit 1
fi

# Switch traffic
echo -e "${YELLOW}Switching traffic to ${NEW_ENV}...${NC}"
kubectl patch svc ${SERVICE_NAME} -n ${NAMESPACE_BLUE} -p "{\"spec\":{\"selector\":{\"app\":\"inference\",\"version\":\"${NEW_ENV}\"}}}"

# Wait for propagation
sleep 5

# Verify switch
NEW_SELECTOR=$(kubectl get svc ${SERVICE_NAME} -n ${NAMESPACE_BLUE} -o jsonpath='{.spec.selector.version}')
echo -e "${GREEN}Traffic now points to: ${NEW_SELECTOR}${NC}"

# Update ingress if needed
if [ "$NEW_ENV" == "green" ]; then
    # For green deployment, we need to update the ingress to point to green's service
    # This is optional if you're using a single service that switches selector
    echo -e "${YELLOW}Updating ingress for green deployment...${NC}"
    # Add ingress update logic if needed
fi

echo -e "${GREEN}=== Switch complete! ===${NC}"
echo -e "${BLUE}Old environment (${CURRENT_SELECTOR}) can now be scaled down or updated${NC}"
