from __future__ import annotations


class CurrentUser:
    def __init__(self):
        self.user_id = "local-user"
        self.email = "dev@example.com"
        self.roles = ["USER", "admin"]
        self.tenant_id = "local_tenant"
        self.tenant_key = "local"
        self.app_id = "app"
        self.employee_code = "dev001"

    def has_role(self, role: str) -> bool:
        return role in self.roles


async def get_current_user() -> CurrentUser:
    return CurrentUser()
