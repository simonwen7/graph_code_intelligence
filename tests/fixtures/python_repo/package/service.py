class PaymentService:
    def charge(self, amount: int) -> bool:
        return amount > 0
