# Enterprise Knowledge Base

## Overview

This knowledge base covers platform tools, engineering standards, and operational
guidelines used across the enterprise. It is used by the Enterprise Copilot to
answer employee questions accurately and consistently.

## Platform Tools

### CI/CD Pipeline

The enterprise uses a GitHub Actions-based CI/CD pipeline for all service
deployments. Every pull request must pass the following checks before merging:

- Unit tests (minimum 80 % coverage required)
- Integration tests
- Static analysis (linting, type checking)
- Security scanning (dependency audit + SAST)
- Container image build and push to registry

Deployment to production requires a manual approval gate from a senior engineer.
Rollbacks are automated: if error-rate exceeds 5 % post-deploy, the system
auto-rolls back within 2 minutes.

### Harness Engineering Platform

Harness is the enterprise's primary deployment and feature-flag management tool.
Engineering teams use Harness for:

- Canary and blue-green deployments
- Feature flags for A/B testing and gradual roll-outs
- Pipeline triggers on merge to main
- Cost management and cloud spend visibility

To onboard a new service into Harness, submit a request via the internal
service-registry portal and attach the Dockerfile and deployment manifest.

### Observability Stack

The observability platform is built on:

- **Metrics**: Prometheus + Grafana (dashboards available at grafana.internal)
- **Logging**: Elasticsearch + Kibana (ELK stack)
- **Tracing**: Jaeger for distributed traces
- **Alerting**: PagerDuty with on-call rotation

All services must emit the following standard metrics:
- `http_request_duration_seconds` (histogram)
- `http_requests_total` (counter)
- `error_rate` (gauge)
- Custom business KPIs specific to the service domain

### Data Platform

The enterprise data platform is managed by the Data Engineering team and consists of:

- **Ingestion**: Apache Kafka for real-time streams, Airbyte for batch connectors
- **Storage**: Delta Lake on Azure ADLS Gen2
- **Transformation**: dbt (data build tool) for modelling and lineage
- **Orchestration**: Apache Airflow (managed via Astronomer)
- **Serving**: Databricks SQL Warehouse for analytics, Snowflake for data sharing
- **Governance**: Apache Atlas for metadata, Unity Catalog for fine-grained access

Data SLAs: Bronze layer within 15 minutes of ingestion, Silver within 2 hours,
Gold within 4 hours.

## AI and Machine Learning Platform

### Model Serving

Models are served via a centralised ML serving layer built on:

- **Online inference**: BentoML + Kubernetes (autoscaling by request rate)
- **Batch inference**: Spark jobs scheduled via Airflow
- **LLM gateway**: Internal proxy that routes requests to OpenAI, Azure OpenAI,
  or on-prem models depending on data residency requirements

Cost optimisation for LLM inference uses intelligent routing:
- Simple classification and extraction tasks → GPT-3.5-turbo or equivalent
- Standard reasoning tasks → GPT-4o-mini or equivalent
- Complex multi-step reasoning, confidential data → Azure OpenAI GPT-4 (on-prem)

All LLM calls are logged for cost tracking. The monthly LLM budget is reviewed
by the AI Platform team.

### Evaluation and Testing (Harness AI Eval)

The AI evaluation framework (also called the "Harness") tests model quality on:

- Accuracy benchmarks (MMLU, HumanEval, domain-specific eval sets)
- Regression test suites run on every model update
- Red-teaming and safety evaluation
- Latency and throughput benchmarking

An "eval harness" in the AI context means the infrastructure that:
1. Runs prompts/tasks against models
2. Collects and scores outputs
3. Compares results across model versions
4. Detects regressions before deployment

### MLOps Workflow

The standard ML development workflow:

1. Data preparation and feature engineering (Feature Store)
2. Experimentation in Databricks notebooks or local environment
3. Training job tracked in MLflow
4. Model registered in MLflow Model Registry
5. Automated evaluation via eval harness
6. Deployment via Harness CD pipeline
7. A/B testing with feature flags
8. Monitoring with Grafana dashboards

## Engineering Standards

### API Design

All internal APIs must follow these standards:

- RESTful design with OpenAPI 3.0 specification
- JSON response bodies with consistent error schema
- Authentication via JWT tokens (issued by internal IdP)
- Rate limiting: 1000 req/min per service account by default
- Versioning: URL path versioning (`/v1/`, `/v2/`)
- Deprecation notice: minimum 6-month window before removing endpoints

### Security

Security requirements for all services:

- Secrets stored in HashiCorp Vault (never in code or environment files)
- Network policies: zero-trust; services communicate only via explicit allow-lists
- Container images must be based on approved base images (see security portal)
- Dependency scanning: automated weekly and on every build
- RBAC: principle of least privilege for all service accounts
- Penetration testing: annually by external vendor
- SIEM: all access logs forwarded to Splunk

### Code Quality

Coding standards across all teams:

- All code must be reviewed by at least one peer (two for security-critical paths)
- Branch protection on `main` and `release/*`
- Commit message format: Conventional Commits (`feat:`, `fix:`, `chore:`, etc.)
- Unit tests required for all business logic
- End-to-end tests for critical user journeys
- Documentation: every service must have a `README.md` and runbook

## HR and People Operations

### Onboarding

New employees receive:

- Laptop and access provisioning (Day 1) via IT service desk
- Buddy programme for first 30 days
- 90-day onboarding plan with manager
- Access to internal LMS (Learning Management System) for mandatory training
- GitHub org invite and SSO setup within first hour

### Learning and Development

The company provides:

- Annual learning budget of $2,000 per employee
- Access to O'Reilly Learning and internal courses
- Conference attendance (approved by manager)
- Internal tech talks every other Friday

### Leave Policy

- Annual leave: 25 days per year
- Sick leave: unlimited (self-certified for <5 days)
- Parental leave: 26 weeks fully paid (primary caregiver), 12 weeks (secondary)
- Leave requests submitted via the HR portal at least 2 weeks in advance

## IT and Infrastructure

### Laptop Setup

Standard development environment:

- MacBook Pro 14" M-series or equivalent Linux workstation
- Homebrew (macOS) for package management
- Docker Desktop for local containers
- VS Code or JetBrains IDEs (both supported)
- VPN: Tailscale (always-on for corporate resources)

### Access Requests

All access requests go through the IAM portal:

- Standard access: auto-approved within 1 business day
- Elevated/admin access: requires manager + security team approval
- External vendor access: requires CISO sign-off

### Incident Management

Severity levels:

- **P0**: Complete outage, all hands. Resolution SLA: 1 hour
- **P1**: Major degradation. Resolution SLA: 4 hours
- **P2**: Minor degradation. Resolution SLA: 24 hours
- **P3**: No user impact, cosmetic. Resolution SLA: 1 week

Incident response: page on-call via PagerDuty, post in #incidents Slack channel,
create incident ticket in Jira.

## Cost Management and FinOps

### Cloud Spend Guidelines

- AWS is the primary cloud; GCP is used for specific AI workloads
- Every team has a monthly budget with alerts at 80 % and 100 % threshold
- Spot/preemptible instances required for non-critical batch workloads
- Reserved instances for stable, long-running services (1-year term)
- Resource tagging mandatory: `team`, `environment`, `cost-centre` tags on all resources

### AI Cost Optimisation

Key strategies used enterprise-wide to reduce AI inference cost:

1. **Intelligent routing**: Route requests to the cheapest model that meets quality requirements
2. **Semantic caching**: Cache LLM responses by semantic similarity; reuse when similarity > 90 %
3. **Context compression**: Trim retrieved chunks to top-3 after reranking; summarise history
4. **Batching**: Aggregate low-priority requests and run in batch mode
5. **Model distillation**: Fine-tune smaller models on high-quality outputs from large models

The AI platform reports monthly: cost per request, cache hit rate, and
quality-adjusted cost per successful outcome.

## Frequently Asked Questions

### How do I request a new service account?

Submit a ticket in Jira under the `PLATFORM` project with the service name, owner,
and required permissions. The Platform team provisions within 2 business days.

### How do I add a new data source to the data platform?

1. Open a request in the Data Engineering intake form (link in the data portal)
2. Provide source system details, data volume, and schema
3. The DE team will schedule an onboarding call within 5 business days
4. Airbyte connector config reviewed and deployed by DE team

### Who do I contact for AI model access?

Contact the AI Platform team via the `#ai-platform` Slack channel or email
ai-platform@company.com. Include your use case and expected request volume.

### What is the process for releasing a new feature?

1. Create a feature branch from `main`
2. Develop and test locally
3. Open a PR with description and test evidence
4. Pass CI checks and get at least one code review approval
5. Merge to `main` (deploys automatically to staging)
6. Test in staging
7. Request production deployment via Harness pipeline (requires approval)
8. Monitor for 30 minutes post-deploy; confirm no regressions

### How does the eval harness work in AI?

The AI eval harness is an automated framework that:
- Runs test prompts against models after every change
- Scores outputs using accuracy metrics and LLM-as-judge evaluation
- Compares scores against a baseline to detect regressions
- Generates reports visible in the AI Platform dashboard

Think of it as a CI pipeline, but for model quality instead of code correctness.
