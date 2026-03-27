# ==============================================================================
# FinSage — Terraform Infrastructure
# Provider  : Databricks (databricks/databricks ~> 1.40)
# Manages   : Job cluster policy, job definition, and SP lookup.
#
# NOTE: In FinSage, the Databricks Asset Bundle (databricks.yml) is the
# authoritative source of truth for job topology and deployment.  This
# Terraform module manages complementary, longer-lived infrastructure
# (cluster policies, secret scopes, cost tags) that falls outside the DAB scope.
# ==============================================================================

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    databricks = {
      source  = "databricks/databricks"
      version = "~> 1.40"
    }
  }

  # ---------------------------------------------------------------------------
  # Remote state — uncomment and configure before running in production.
  # ---------------------------------------------------------------------------
  # backend "azurerm" {
  #   resource_group_name  = "finsage-tfstate-rg"
  #   storage_account_name = "finsagetfstate"
  #   container_name       = "tfstate"
  #   key                  = "finsage/${var.environment}.tfstate"
  # }
}

# ---------------------------------------------------------------------------
# Provider
# Authenticates via environment variables set by CI (same as the DAB workflow):
#   DATABRICKS_HOST, DATABRICKS_CLIENT_ID, DATABRICKS_CLIENT_SECRET
# ---------------------------------------------------------------------------
provider "databricks" {}

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------
#variable "databricks_host" {
#  description = "Databricks workspace URL (e.g. https://dbc-xxxx.cloud.databricks.com)"
#  type        = string
#}

variable "environment" {
  description = "Deployment environment label: dev or prod"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "prod"], var.environment)
    error_message = "environment must be 'dev' or 'prod'."
  }
}

variable "node_type_id" {
  description = "EC2 instance type for the job cluster"
  type        = string
  default     = "i3.xlarge"
}

variable "notification_email" {
  description = "Email address for job failure alerts"
  type        = string
  default     = "digvijay@arsaga.jp"
}

# ---------------------------------------------------------------------------
# Data source — resolve the service principal that will run the job
# ---------------------------------------------------------------------------
data "databricks_service_principal" "finsage_sp" {
  display_name = "finsage-service-principal"
}

# ---------------------------------------------------------------------------
# Cluster policy — enforces cost guardrails on all FinSage clusters.
# The DAB job references this policy by ID via a library config (future work).
# ---------------------------------------------------------------------------
resource "databricks_cluster_policy" "finsage_policy" {
  name = "finsage-${var.environment}-policy"

  definition = jsonencode({
    "spark_version" = {
      type  = "fixed"
      value = "14.3.x-scala2.12"
    }
    "num_workers" = {
      type         = "range"
      minValue     = 1
      maxValue     = 4
      defaultValue = 2
    }
    "autotermination_minutes" = {
      type         = "fixed"
      value        = 30
    }
    "custom_tags.project" = {
      type  = "fixed"
      value = "finsage"
    }
    "custom_tags.managed_by" = {
      type  = "fixed"
      value = "terraform"
    }
    "custom_tags.env" = {
      type  = "fixed"
      value = var.environment
    }
  })
}

# ---------------------------------------------------------------------------
# Secret scope — stores runtime secrets accessible to notebook jobs.
# Secrets are populated manually or via a secrets management pipeline.
# ---------------------------------------------------------------------------
resource "databricks_secret_scope" "finsage_secrets" {
  name = "finsage-${var.environment}"
}

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
output "cluster_policy_id" {
  description = "Databricks cluster policy ID — reference this in databricks.yml policy_id"
  value       = databricks_cluster_policy.finsage_policy.id
}

output "secret_scope_name" {
  description = "Databricks secret scope name for runtime secrets"
  value       = databricks_secret_scope.finsage_secrets.name
}

output "service_principal_app_id" {
  description = "Application (client) ID of the FinSage service principal"
  value       = data.databricks_service_principal.finsage_sp.application_id
  sensitive   = true
}
