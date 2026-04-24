"""Central exception taxonomy for claude_teams.

Each class owns its own message template so call sites stay short (TRY003)
while preserving the exact text downstream tests string-match against.

Inheritance follows the base class the original callsite raised
(``ValueError``/``TypeError``/``RuntimeError``/``KeyError``/
``FileNotFoundError``/``ToolError``) so ``except`` clauses elsewhere keep
matching without modification.
"""

from collections.abc import Iterable

from fastmcp.exceptions import ToolError

# ---------------------------------------------------------------------------
# ValueError subclasses
# ---------------------------------------------------------------------------


class TaskSubjectEmptyError(ValueError):
    """Raised when a task subject is missing or whitespace-only."""

    def __init__(self) -> None:
        """Build the fixed task-subject-empty message."""
        super().__init__("Task subject must not be empty")


class TeamNotFoundValueError(ValueError):
    """Raised when a referenced team does not exist (domain layer)."""

    def __init__(self, team_name: str) -> None:
        """Build the message from the missing team name."""
        super().__init__(f"Team {team_name!r} does not exist")


class TaskSelfBlockError(ValueError):
    """Raised when a task attempts to block itself."""

    def __init__(self, task_id: str) -> None:
        """Build the message from the offending task identifier."""
        super().__init__(f"Task {task_id} cannot block itself")


class TaskSelfBlockedByError(ValueError):
    """Raised when a task attempts to be blocked by itself."""

    def __init__(self, task_id: str) -> None:
        """Build the message from the offending task identifier."""
        super().__init__(f"Task {task_id} cannot be blocked by itself")


class TaskReferenceNotFoundError(ValueError):
    """Raised when a referenced task id does not exist."""

    def __init__(self, referenced_id: str) -> None:
        """Build the message from the missing referenced task id."""
        super().__init__(f"Referenced task {referenced_id!r} does not exist")


class CyclicTaskBlockError(ValueError):
    """Raised when adding a ``blocks`` edge would create a cycle."""

    def __init__(self, task_id: str, blocked_id: str) -> None:
        """Build the message from the source and target task identifiers."""
        super().__init__(
            f"Adding block {task_id} -> {blocked_id} would create a circular dependency"
        )


class CyclicTaskBlockedByError(ValueError):
    """Raised when adding a ``blocked_by`` edge would create a cycle."""

    def __init__(self, task_id: str, blocker_id: str) -> None:
        """Build the message from the dependent and blocker task identifiers."""
        super().__init__(
            f"Adding dependency {task_id} blocked_by {blocker_id} "
            "would create a circular dependency"
        )


class InvalidTaskStatusError(ValueError):
    """Raised when a task status value is not recognized."""

    def __init__(self, new_status: str) -> None:
        """Build the message from the rejected status value."""
        super().__init__(f"Invalid status: {new_status!r}")


class TaskStatusRegressionError(ValueError):
    """Raised when a task status transition moves backwards."""

    def __init__(self, current_status: str, new_status: str) -> None:
        """Build the message from the current and requested statuses."""
        super().__init__(f"Cannot transition from {current_status!r} to {new_status!r}")


class BlockedTaskStatusError(ValueError):
    """Raised when a task advances status while blockers are incomplete."""

    def __init__(self, new_status: str, blocker_id: str, blocker_status: str) -> None:
        """Build the message from the target status and outstanding blocker."""
        super().__init__(
            f"Cannot set status to {new_status!r}: "
            f"blocked by task {blocker_id} (status: {blocker_status!r})"
        )


class InvalidNameError(ValueError):
    """Raised when a filesystem-unsafe identifier is supplied."""

    def __init__(self, label: str, name: str) -> None:
        """Build the message from the field label and rejected name."""
        super().__init__(
            f"Invalid {label}: {name!r}. "
            "Use only letters, numbers, hyphens, underscores."
        )


class NameTooLongError(ValueError):
    """Raised when an identifier exceeds the configured length cap."""

    def __init__(self, label: str, name: str, max_len: int) -> None:
        """Build the message from the label, offending name, and maximum length."""
        super().__init__(
            f"{label.capitalize()} too long ({len(name)} chars, "
            f"max {max_len}): {name[:20]!r}..."
        )


class MemberAlreadyExistsError(ValueError):
    """Raised when a teammate name collides with an existing member."""

    def __init__(self, member_name: str, team_name: str) -> None:
        """Build the message from the member name and team name."""
        super().__init__(f"Member {member_name!r} already exists in team {team_name!r}")


class TeamAlreadyExistsError(ValueError):
    """Raised when team creation targets a name that already has a config on disk.

    The domain layer raises this so the callers (MCP bootstrap, CLI,
    preset expansion) can decide how to surface the collision without
    silently overwriting an in-flight team.
    """

    def __init__(self, team_name: str) -> None:
        """Build the message from the colliding team name."""
        super().__init__(f"Team {team_name!r} already exists")


class PresetEmptyMembersError(ValueError):
    """Raised when a preset is registered without any ``PresetMemberSpec`` entries.

    A zero-member preset would expand to a team with no teammates,
    which collapses the whole point of the preset composition pattern.
    Surface the mistake at registration rather than at expansion time
    so plugin authors fail fast.
    """

    def __init__(self, preset_name: str) -> None:
        """Build the message from the offending preset name."""
        super().__init__(
            f"Preset {preset_name!r} has no members; presets must declare "
            "at least one PresetMemberSpec to be useful."
        )


class CannotRemoveLeadError(ValueError):
    """Raised when someone tries to remove the reserved ``team-lead``."""

    def __init__(self) -> None:
        """Build the fixed team-lead removal message."""
        super().__init__("Cannot remove team-lead from team")


class TaskAssignmentNoOwnerError(ValueError):
    """Raised when a task assignment message is missing an owner."""

    def __init__(self) -> None:
        """Build the fixed missing-owner message."""
        super().__init__("Cannot send task assignment: task has no owner")


class InvalidInboxOffsetError(ValueError):
    """Raised when an inbox pagination offset is negative."""

    def __init__(self) -> None:
        """Build the fixed negative-offset message."""
        super().__init__("offset must be >= 0")


class InvalidInboxOrderError(ValueError):
    """Raised when an inbox ordering value is not recognized."""

    def __init__(self) -> None:
        """Build the fixed invalid-order message."""
        super().__init__("order must be 'oldest' or 'newest'")


class InvalidEnvVarNameError(ValueError):
    """Raised when a spawn env-var key fails the safe-name pattern."""

    def __init__(self, key: str) -> None:
        """Build the message from the rejected environment variable name."""
        super().__init__(f"Invalid environment variable name: {key!r}")


class UnsupportedBackendModelError(ValueError):
    """Raised when a model name is not recognized for a backend."""

    def __init__(
        self, generic_name: str, backend_name: str, supported: Iterable[str]
    ) -> None:
        """Build the message from the rejected model and the supported list."""
        super().__init__(
            f"Unsupported model {generic_name!r} for {backend_name}. "
            f"Supported: {', '.join(supported)}"
        )


class PermissionBypassUnsupportedValueError(ValueError):
    """Raised when ``permission_mode='bypass'`` is requested without support."""

    def __init__(self, backend_name: str) -> None:
        """Build the message from the backend that rejects bypass mode."""
        super().__init__(
            f"Backend {backend_name!r} does not support permission_mode='bypass'."
        )


# ---------------------------------------------------------------------------
# RuntimeError subclasses
# ---------------------------------------------------------------------------


class InboxEncryptionKeyMissingError(RuntimeError):
    """Raised when an encrypted inbox is accessed without a master key."""

    def __init__(self, env_var: str) -> None:
        """Build the message from the expected master-key env var name."""
        super().__init__(
            f"Inbox encryption key is required for this inbox, "
            f"but {env_var} is not set."
        )


class InboxDecryptError(RuntimeError):
    """Raised when a stored inbox entry fails to decrypt."""

    def __init__(self) -> None:
        """Build the fixed decrypt-failure message."""
        super().__init__(
            "Unable to decrypt inbox entry with the configured encryption key."
        )


class InboxMasterKeyTooShortError(RuntimeError):
    """Raised when the configured inbox master key is below the minimum length."""

    def __init__(self, env_var: str, min_len: int) -> None:
        """Build the message from the env var name and required minimum length."""
        super().__init__(
            f"{env_var} must be at least {min_len} characters "
            "to provide adequate entropy for inbox encryption."
        )


class NoBackendsAvailableError(RuntimeError):
    """Raised when no spawner backend binaries are on PATH."""

    def __init__(self) -> None:
        """Build the fixed no-backends message."""
        super().__init__(
            "No backends available. Install at least one agentic CLI tool."
        )


class TmuxPaneCreationError(RuntimeError):
    """Raised when tmux fails to create a pane for a spawn request."""

    def __init__(self, agent_name: str) -> None:
        """Build the message from the agent whose pane failed to launch."""
        super().__init__(
            f"Failed to create tmux pane for agent {agent_name!r}. "
            "Ensure tmux is running and tmux-cli is available."
        )


class TeamHasMembersError(RuntimeError):
    """Raised when deleting a team that still has non-lead members."""

    def __init__(self, team_name: str, non_lead_count: int) -> None:
        """Build the message from the team name and lingering member count."""
        super().__init__(
            f"Cannot delete team {team_name!r}: "
            f"{non_lead_count} non-lead member(s) still present. "
            "Remove all teammates before deleting."
        )


# ---------------------------------------------------------------------------
# TypeError subclasses
# ---------------------------------------------------------------------------


class MalformedEncryptedInboxEntryError(TypeError):
    """Raised when a stored encrypted inbox entry lacks a ciphertext token."""

    def __init__(self) -> None:
        """Build the fixed malformed-entry message."""
        super().__init__("Malformed encrypted inbox entry: missing ciphertext token.")


class DecryptedInboxNotObjectError(TypeError):
    """Raised when a decrypted inbox entry is not a JSON object."""

    def __init__(self) -> None:
        """Build the fixed non-object message."""
        super().__init__("Decrypted inbox entry must be a JSON object.")


# ---------------------------------------------------------------------------
# KeyError / FileNotFoundError subclasses
# ---------------------------------------------------------------------------


class BackendNotRegisteredError(KeyError):
    """Raised when a requested backend name is not in the registry."""

    def __init__(self, name: str, available: Iterable[str]) -> None:
        """Build the message from the missing name and currently registered list."""
        available_str = ", ".join(sorted(available)) or "(none)"
        super().__init__(f"Backend {name!r} not found. Available: {available_str}")


class CapabilityStoreNotFoundError(FileNotFoundError):
    """Raised when a team has no capability store on disk."""

    def __init__(self, team_name: str) -> None:
        """Build the message from the team whose capability store is missing."""
        super().__init__(f"Capability store not found for team {team_name!r}")


class BackendBinaryNotFoundError(FileNotFoundError):
    """Raised when a backend's CLI binary cannot be located on PATH."""

    def __init__(self, binary_name: str, backend_name: str) -> None:
        """Build the message from the missing binary and owning backend name."""
        super().__init__(
            f"Could not find {binary_name!r} on PATH. "
            f"Install {backend_name} or add it to PATH."
        )


# ---------------------------------------------------------------------------
# ToolError subclasses (MCP-facing)
# ---------------------------------------------------------------------------


class SessionActiveTeamError(ToolError):
    """Raised when a session already has a different active team."""

    def __init__(self, active_team: object) -> None:
        """Build the message from the already-attached team name."""
        super().__init__(
            f"Session already has active team: {active_team}. One team per session."
        )


class InvalidCapabilityError(ToolError):
    """Raised when a team-attach capability token is rejected."""

    def __init__(self) -> None:
        """Build the fixed invalid-capability message."""
        super().__init__("Invalid capability for team attachment.")


class TeamNotFoundToolError(ToolError):
    """Raised when a team lookup fails (MCP-facing)."""

    def __init__(self, team_name: str) -> None:
        """Build the message from the missing team name."""
        super().__init__(f"Team {team_name!r} not found")


class TeamAlreadyExistsToolError(ToolError):
    """Raised when team creation targets an existing team (MCP-facing).

    Mirrors ``TeamNotFoundToolError``'s convention so both ends of the
    team-lookup lifecycle surface a typed ToolError the MCP client can
    distinguish from generic backend errors.
    """

    def __init__(self, team_name: str) -> None:
        """Build the message from the colliding team name."""
        super().__init__(
            f"Team {team_name!r} already exists. Choose a new name or "
            f"delete the existing team first."
        )


class CwdNotAbsoluteError(ToolError):
    """Raised when a spawn ``cwd`` argument is not absolute."""

    def __init__(self, cwd: str) -> None:
        """Build the message from the offending cwd string."""
        super().__init__(f"cwd must be an absolute path, got: {cwd!r}")


class CwdMissingError(ToolError):
    """Raised when a spawn ``cwd`` path does not exist."""

    def __init__(self, candidate: object) -> None:
        """Build the message from the resolved-but-missing cwd."""
        super().__init__(f"cwd does not exist: {candidate}")


class CwdNotDirectoryError(ToolError):
    """Raised when a spawn ``cwd`` path is not a directory."""

    def __init__(self, candidate: object) -> None:
        """Build the message from the non-directory cwd path."""
        super().__init__(f"cwd is not a directory: {candidate}")


class InvalidPermissionModeError(ToolError):
    """Raised when a permission mode argument is not recognized."""

    def __init__(self, raw: str, supported: Iterable[str]) -> None:
        """Build the message from the rejected mode and the accepted set."""
        supported_str = ", ".join(sorted(supported))
        super().__init__(
            f"Invalid permission mode {raw!r}. Supported values: {supported_str}."
        )


class InvalidEnvVarValueError(ToolError):
    """Raised when an env-var-driven default cannot be coerced to the target type.

    Fires from the ``CLAUDE_TEAMS_DEFAULT_*`` precedence layer when an operator
    sets an env var to a value the resolver cannot accept (e.g. non-boolean text
    for a boolean field). Surfaced as ``ToolError`` because the failure happens
    at tool-entry while resolving the precedence chain.
    """

    def __init__(self, env_var: str, raw: str, hint: str) -> None:
        """Build the message from the env-var name, bad value, and remediation hint."""
        super().__init__(f"Invalid value for {env_var}={raw!r}: {hint}.")


class PaginationLimitTooSmallError(ToolError):
    """Raised when a pagination ``limit`` is less than 1."""

    def __init__(self) -> None:
        """Build the fixed too-small-limit message."""
        super().__init__("limit must be >= 1")


class PaginationLimitTooLargeError(ToolError):
    """Raised when a pagination ``limit`` exceeds the cap."""

    def __init__(self, max_page_size: int) -> None:
        """Build the message from the configured maximum page size."""
        super().__init__(f"limit must be <= {max_page_size}")


class PaginationOffsetNegativeError(ToolError):
    """Raised when a pagination ``offset`` is negative."""

    def __init__(self) -> None:
        """Build the fixed negative-offset message."""
        super().__init__("offset must be >= 0")


class AuthenticationRequiredError(ToolError):
    """Raised when a tool requires an authenticated principal."""

    def __init__(self) -> None:
        """Build the fixed authentication-required message."""
        super().__init__(
            "This action requires an attached team session or a valid capability."
        )


class LeadCapabilityRequiredError(ToolError):
    """Raised when a tool requires the team-lead capability."""

    def __init__(self) -> None:
        """Build the fixed lead-capability-required message."""
        super().__init__("This action requires the team-lead capability.")


class PrincipalActingAsOtherError(ToolError):
    """Raised when an authenticated principal claims a different sender name."""

    def __init__(self, principal_name: str, sender: str) -> None:
        """Build the message from the principal name and the claimed sender."""
        super().__init__(
            f"Authenticated principal {principal_name!r} cannot act as {sender!r}."
        )


class PermissionBypassUnsupportedToolError(ToolError):
    """Raised when ``permission_mode='bypass'`` is requested without support."""

    def __init__(self, backend_name: str) -> None:
        """Build the message from the backend that rejects bypass mode."""
        super().__init__(
            f"Backend {backend_name!r} does not support permission_mode='bypass'."
        )


class ReasoningEffortUnsupportedToolError(ToolError):
    """Raised when ``reasoning_effort`` is set for a backend that lacks the dial."""

    def __init__(self, backend_name: str) -> None:
        """Build the message from the backend that does not expose effort."""
        super().__init__(
            f"Backend {backend_name!r} does not accept a reasoning_effort value."
        )


class InvalidReasoningEffortToolError(ToolError):
    """Raised when a ``reasoning_effort`` value is outside a backend's accepted set."""

    def __init__(self, backend_name: str, value: str, supported: Iterable[str]) -> None:
        """Build the message from the backend, rejected value, and valid options."""
        supported_str = ", ".join(sorted(supported))
        super().__init__(
            f"Invalid reasoning_effort {value!r} for {backend_name}. "
            f"Supported: {supported_str}"
        )


class AgentSelectUnsupportedToolError(ToolError):
    """Raised when ``agent_profile`` is set for a backend that lacks selection."""

    def __init__(self, backend_name: str) -> None:
        """Build the message from the backend that does not support selection."""
        super().__init__(
            f"Backend {backend_name!r} does not support agent profile selection."
        )


class UnknownAgentProfileToolError(ToolError):
    """Raised when an ``agent_profile`` value is not among discovered profiles."""

    def __init__(self, backend_name: str, value: str, supported: Iterable[str]) -> None:
        """Build the message from the backend, rejected value, and discovered names."""
        supported_str = ", ".join(sorted(supported)) or "(none discovered)"
        super().__init__(
            f"Unknown agent profile {value!r} for {backend_name}. "
            f"Available: {supported_str}"
        )


class ReservedAgentNameError(ToolError):
    """Raised when a spawn request uses the reserved ``team-lead`` name."""

    def __init__(self) -> None:
        """Build the fixed reserved-name message."""
        super().__init__("Agent name 'team-lead' is reserved")


class BackendSpawnFailedError(ToolError):
    """Raised when a backend spawn raises an unexpected exception."""

    def __init__(self, inner: BaseException) -> None:
        """Build the message from the underlying backend failure."""
        super().__init__(f"Backend spawn failed: {inner}")


class MessageContentEmptyToolError(ToolError):
    """Raised when a plain-message payload has empty body content."""

    def __init__(self) -> None:
        """Build the fixed empty-message-content message."""
        super().__init__("Message content must not be empty")


class MessageSummaryEmptyToolError(ToolError):
    """Raised when a plain-message payload has empty summary text."""

    def __init__(self) -> None:
        """Build the fixed empty-message-summary message."""
        super().__init__("Message summary must not be empty")


class MessageRecipientEmptyToolError(ToolError):
    """Raised when a plain-message payload has no recipient."""

    def __init__(self) -> None:
        """Build the fixed empty-message-recipient message."""
        super().__init__("Message recipient must not be empty")


class BroadcastSummaryEmptyToolError(ToolError):
    """Raised when a broadcast payload has no summary text."""

    def __init__(self) -> None:
        """Build the fixed empty-broadcast-summary message."""
        super().__init__("Broadcast summary must not be empty")


class ShutdownRecipientEmptyToolError(ToolError):
    """Raised when a shutdown request has no recipient."""

    def __init__(self) -> None:
        """Build the fixed empty-shutdown-recipient message."""
        super().__init__("Shutdown request recipient must not be empty")


class PlanRecipientEmptyToolError(ToolError):
    """Raised when a plan-approval response has no recipient."""

    def __init__(self) -> None:
        """Build the fixed empty-plan-recipient message."""
        super().__init__("Plan approval recipient must not be empty")


class NotTeamMemberError(ToolError):
    """Raised when a named principal is not a member of the target team."""

    def __init__(self, role_label: str, name: str, team_name: str) -> None:
        """Build the message from the role label, principal name, and team."""
        super().__init__(f"{role_label} {name!r} is not a member of team {team_name!r}")


class BroadcastSenderError(ToolError):
    """Raised when a broadcast sender is not ``team-lead``."""

    def __init__(self) -> None:
        """Build the fixed broadcast-sender message."""
        super().__init__("Broadcast sender must be 'team-lead'")


class ShutdownSelfError(ToolError):
    """Raised when a shutdown request targets ``team-lead``."""

    def __init__(self) -> None:
        """Build the fixed self-shutdown message."""
        super().__init__("Cannot send shutdown request to team-lead")


class ShutdownResponseApprovalRequiredError(ToolError):
    """Raised when a shutdown response omits an explicit approve/reject flag."""

    def __init__(self) -> None:
        """Build the fixed missing-approve message."""
        super().__init__(
            "shutdown_response requires an explicit approve=true|false flag"
        )


class BroadcastTooManyRecipientsError(ToolError):
    """Raised when a broadcast would fan out past the allowed recipient cap."""

    def __init__(self, count: int, limit: int) -> None:
        """Build the message from the attempted recipient count and the cap."""
        super().__init__(
            f"Broadcast recipient count {count} exceeds the maximum of {limit}; "
            "send targeted messages instead."
        )


class UnknownMessageTypeError(ToolError):
    """Raised when ``send_message`` dispatch sees an unsupported type."""

    def __init__(self, message_type: str) -> None:
        """Build the message from the unrecognized message type tag."""
        super().__init__(f"Unknown message type: {message_type}")


class TaskNotFoundToolError(ToolError):
    """Raised when a task lookup fails (MCP-facing)."""

    def __init__(self, task_id: str, team_name: str) -> None:
        """Build the message from the missing task id and owning team."""
        super().__init__(f"Task {task_id!r} not found in team {team_name!r}")


class InboxAccessDeniedError(ToolError):
    """Raised when a principal cannot read or poll another agent's inbox."""

    def __init__(self, action: str, principal_name: str, agent_name: str) -> None:
        """Build the message from the action verb, principal, and target inbox."""
        super().__init__(
            f"Authenticated principal {principal_name!r} "
            f"cannot {action} inbox {agent_name!r}."
        )


class TeammateNotFoundToolError(ToolError):
    """Raised when a teammate lookup in a team fails (MCP-facing)."""

    def __init__(self, agent_name: str, team_name: str) -> None:
        """Build the message from the missing teammate name and team."""
        super().__init__(f"Teammate {agent_name!r} not found in team {team_name!r}")


class ShutdownLeadError(ToolError):
    """Raised when a shutdown approval targets ``team-lead``."""

    def __init__(self) -> None:
        """Build the fixed cannot-shutdown-lead message."""
        super().__init__("Cannot process shutdown for team-lead")


class NoProcessHandleError(ToolError):
    """Raised when a teammate has no recorded process handle."""

    def __init__(self, agent_name: str) -> None:
        """Build the message from the teammate lacking a process handle."""
        super().__init__(f"No process handle for teammate {agent_name!r}")


class UnknownTemplateToolError(ToolError):
    """Raised when ``options.template`` names an unregistered template."""

    def __init__(self, value: str, available: Iterable[str]) -> None:
        """Build the message from the rejected name and registered templates."""
        available_str = ", ".join(sorted(available)) or "(none registered)"
        super().__init__(f"Unknown template {value!r}. Available: {available_str}")


class UnknownPresetToolError(ToolError):
    """Raised when ``create_team_from_preset`` names an unregistered preset."""

    def __init__(self, value: str, available: Iterable[str]) -> None:
        """Build the message from the rejected name and registered presets."""
        available_str = ", ".join(sorted(available)) or "(none registered)"
        super().__init__(f"Unknown preset {value!r}. Available: {available_str}")


class PresetMemberSpawnFailedError(ToolError):
    """Raised when a preset expansion fails partway through fan-out.

    The team and any already-spawned members persist — expansion is
    intentionally non-transactional so callers can retry the remaining
    members or tear the team down with ``team_delete``.
    """

    def __init__(self, member_name: str, cause: BaseException) -> None:
        """Build the message from the failing member and root cause.

        Includes the cause's class name so log consumers that only see
        the outer message can tell a template-resolution failure apart
        from a backend-spawn failure without chasing ``__cause__``.
        """
        super().__init__(
            f"Preset expansion failed on member {member_name!r} "
            f"({type(cause).__name__}): {cause}"
        )
