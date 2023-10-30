{
    "routing": "DUTCH_LIMIT",
    "quote": {
        "orderInfo": {
            "chainId": 1,
            "permit2Address": "0x000000000022d473030f116ddee9f6b43ac78ba3",
            "reactor": "0x6000da47483062A0D734Ba3dc7576Ce6A0B645C4",
            "swapper": "0x0000000000000000000000000000000000000000",
            "nonce": "1993352615960370069291494398936148182539234688673283608969519365017077951233",
            "deadline": 1698670834,
            "additionalValidationContract": "0x0000000000000000000000000000000000000000",
            "additionalValidationData": "0x",
            "decayStartTime": 1698670762,
            "decayEndTime": 1698670822,
            "exclusiveFiller": "0x32801aB1957Aaad1c65289B51603373802B4e8BB",
            "exclusivityOverrideBps": "100",
            "input": {
                "token": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
                "startAmount": "100000000000000000000",
                "endAmount": "100000000000000000000",
            },
            "outputs": [
                {
                    "token": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                    "startAmount": "50732873479533414",
                    "endAmount": "40507249443929631",
                    "recipient": "0x0000000000000000000000000000000000000000",
                },
                {
                    "token": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                    "startAmount": "76213630665298",
                    "endAmount": "60852152394486",
                    "recipient": "0x37a8f295612602f2774d331e562be9e61B83a327",
                },
            ],
        },
        "encodedOrder": "0x0000000000000000000000000000000000000000000000000000000000000020000000000000000000000000000000000000000000000000000000000000012000000000000000000000000000000000000000000000000000000000653fa8aa00000000000000000000000000000000000000000000000000000000653fa8e600000000000000000000000032801ab1957aaad1c65289b51603373802b4e8bb00000000000000000000000000000000000000000000000000000000000000640000000000000000000000006b175474e89094c44da98b954eedeac495271d0f0000000000000000000000000000000000000000000000056bc75e2d631000000000000000000000000000000000000000000000000000056bc75e2d6310000000000000000000000000000000000000000000000000000000000000000002000000000000000000000000006000da47483062a0d734ba3dc7576ce6a0b645c4000000000000000000000000000000000000000000000000000000000000000004683295d60a099aaf9f1be560f73109eeaba6d826fe487999d1e8df70362b0100000000000000000000000000000000000000000000000000000000653fa8f2000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000c000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000002000000000000000000000000c02aaa39b223fe8d0a0e5c4f27ead9083c756cc200000000000000000000000000000000000000000000000000b43d47962eb366000000000000000000000000000000000000000000000000008fe920f5eefe1f0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000c02aaa39b223fe8d0a0e5c4f27ead9083c756cc200000000000000000000000000000000000000000000000000004550de620252000000000000000000000000000000000000000000000000000037583eed62f600000000000000000000000037a8f295612602f2774d331e562be9e61b83a327",
        "quoteId": "93630d29-3118-4231-8bbd-6383a17e60e3",
        "requestId": "2c5bc13a-e1af-458c-98ea-9b639630a7ea",
        "orderHash": "0xa01b1b785cd44e3402117f361fdd7a36f6e7e1b0b8e6d8779cbec2c423d5e14c",
        "startTimeBufferSecs": 45,
        "auctionPeriodSecs": 60,
        "deadlineBufferSecs": 12,
        "slippageTolerance": "0.5",
        "permitData": {
            "domain": {
                "name": "Permit2",
                "chainId": 1,
                "verifyingContract": "0x000000000022d473030f116ddee9f6b43ac78ba3",
            },
            "types": {
                "PermitWitnessTransferFrom": [
                    {"name": "permitted", "type": "TokenPermissions"},
                    {"name": "spender", "type": "address"},
                    {"name": "nonce", "type": "uint256"},
                    {"name": "deadline", "type": "uint256"},
                    {"name": "witness", "type": "ExclusiveDutchOrder"},
                ],
                "TokenPermissions": [
                    {"name": "token", "type": "address"},
                    {"name": "amount", "type": "uint256"},
                ],
                "ExclusiveDutchOrder": [
                    {"name": "info", "type": "OrderInfo"},
                    {"name": "decayStartTime", "type": "uint256"},
                    {"name": "decayEndTime", "type": "uint256"},
                    {"name": "exclusiveFiller", "type": "address"},
                    {"name": "exclusivityOverrideBps", "type": "uint256"},
                    {"name": "inputToken", "type": "address"},
                    {"name": "inputStartAmount", "type": "uint256"},
                    {"name": "inputEndAmount", "type": "uint256"},
                    {"name": "outputs", "type": "DutchOutput[]"},
                ],
                "OrderInfo": [
                    {"name": "reactor", "type": "address"},
                    {"name": "swapper", "type": "address"},
                    {"name": "nonce", "type": "uint256"},
                    {"name": "deadline", "type": "uint256"},
                    {"name": "additionalValidationContract", "type": "address"},
                    {"name": "additionalValidationData", "type": "bytes"},
                ],
                "DutchOutput": [
                    {"name": "token", "type": "address"},
                    {"name": "startAmount", "type": "uint256"},
                    {"name": "endAmount", "type": "uint256"},
                    {"name": "recipient", "type": "address"},
                ],
            },
            "values": {
                "permitted": {
                    "token": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
                    "amount": {"type": "BigNumber", "hex": "0x056bc75e2d63100000"},
                },
                "spender": "0x6000da47483062A0D734Ba3dc7576Ce6A0B645C4",
                "nonce": {
                    "type": "BigNumber",
                    "hex": "0x04683295d60a099aaf9f1be560f73109eeaba6d826fe487999d1e8df70362b01",
                },
                "deadline": 1698670834,
                "witness": {
                    "info": {
                        "reactor": "0x6000da47483062A0D734Ba3dc7576Ce6A0B645C4",
                        "swapper": "0x0000000000000000000000000000000000000000",
                        "nonce": {
                            "type": "BigNumber",
                            "hex": "0x04683295d60a099aaf9f1be560f73109eeaba6d826fe487999d1e8df70362b01",
                        },
                        "deadline": 1698670834,
                        "additionalValidationContract": "0x0000000000000000000000000000000000000000",
                        "additionalValidationData": "0x",
                    },
                    "decayStartTime": 1698670762,
                    "decayEndTime": 1698670822,
                    "exclusiveFiller": "0x32801aB1957Aaad1c65289B51603373802B4e8BB",
                    "exclusivityOverrideBps": {"type": "BigNumber", "hex": "0x64"},
                    "inputToken": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
                    "inputStartAmount": {
                        "type": "BigNumber",
                        "hex": "0x056bc75e2d63100000",
                    },
                    "inputEndAmount": {
                        "type": "BigNumber",
                        "hex": "0x056bc75e2d63100000",
                    },
                    "outputs": [
                        {
                            "token": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                            "startAmount": {
                                "type": "BigNumber",
                                "hex": "0xb43d47962eb366",
                            },
                            "endAmount": {
                                "type": "BigNumber",
                                "hex": "0x8fe920f5eefe1f",
                            },
                            "recipient": "0x0000000000000000000000000000000000000000",
                        },
                        {
                            "token": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                            "startAmount": {
                                "type": "BigNumber",
                                "hex": "0x4550de620252",
                            },
                            "endAmount": {"type": "BigNumber", "hex": "0x37583eed62f6"},
                            "recipient": "0x37a8f295612602f2774d331e562be9e61B83a327",
                        },
                    ],
                },
            },
        },
        "portionBips": 15,
        "portionAmount": "76213630665298",
        "portionRecipient": "0x37a8f295612602f2774d331e562be9e61B83a327",
    },
    "requestId": "2c5bc13a-e1af-458c-98ea-9b639630a7ea",
    "allQuotes": [
        {
            "routing": "DUTCH_LIMIT",
            "quote": {
                "orderInfo": {
                    "chainId": 1,
                    "permit2Address": "0x000000000022d473030f116ddee9f6b43ac78ba3",
                    "reactor": "0x6000da47483062A0D734Ba3dc7576Ce6A0B645C4",
                    "swapper": "0x0000000000000000000000000000000000000000",
                    "nonce": "1993352615960370069291494398936148182539234688673283608969519365017077951233",
                    "deadline": 1698670834,
                    "additionalValidationContract": "0x0000000000000000000000000000000000000000",
                    "additionalValidationData": "0x",
                    "decayStartTime": 1698670762,
                    "decayEndTime": 1698670822,
                    "exclusiveFiller": "0x32801aB1957Aaad1c65289B51603373802B4e8BB",
                    "exclusivityOverrideBps": "100",
                    "input": {
                        "token": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
                        "startAmount": "100000000000000000000",
                        "endAmount": "100000000000000000000",
                    },
                    "outputs": [
                        {
                            "token": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                            "startAmount": "50732873479533414",
                            "endAmount": "40507249443929631",
                            "recipient": "0x0000000000000000000000000000000000000000",
                        },
                        {
                            "token": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                            "startAmount": "76213630665298",
                            "endAmount": "60852152394486",
                            "recipient": "0x37a8f295612602f2774d331e562be9e61B83a327",
                        },
                    ],
                },
                "encodedOrder": "0x0000000000000000000000000000000000000000000000000000000000000020000000000000000000000000000000000000000000000000000000000000012000000000000000000000000000000000000000000000000000000000653fa8aa00000000000000000000000000000000000000000000000000000000653fa8e600000000000000000000000032801ab1957aaad1c65289b51603373802b4e8bb00000000000000000000000000000000000000000000000000000000000000640000000000000000000000006b175474e89094c44da98b954eedeac495271d0f0000000000000000000000000000000000000000000000056bc75e2d631000000000000000000000000000000000000000000000000000056bc75e2d6310000000000000000000000000000000000000000000000000000000000000000002000000000000000000000000006000da47483062a0d734ba3dc7576ce6a0b645c4000000000000000000000000000000000000000000000000000000000000000004683295d60a099aaf9f1be560f73109eeaba6d826fe487999d1e8df70362b0100000000000000000000000000000000000000000000000000000000653fa8f2000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000c000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000002000000000000000000000000c02aaa39b223fe8d0a0e5c4f27ead9083c756cc200000000000000000000000000000000000000000000000000b43d47962eb366000000000000000000000000000000000000000000000000008fe920f5eefe1f0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000c02aaa39b223fe8d0a0e5c4f27ead9083c756cc200000000000000000000000000000000000000000000000000004550de620252000000000000000000000000000000000000000000000000000037583eed62f600000000000000000000000037a8f295612602f2774d331e562be9e61b83a327",
                "quoteId": "93630d29-3118-4231-8bbd-6383a17e60e3",
                "requestId": "2c5bc13a-e1af-458c-98ea-9b639630a7ea",
                "orderHash": "0xa01b1b785cd44e3402117f361fdd7a36f6e7e1b0b8e6d8779cbec2c423d5e14c",
                "startTimeBufferSecs": 45,
                "auctionPeriodSecs": 60,
                "deadlineBufferSecs": 12,
                "slippageTolerance": "0.5",
                "permitData": {
                    "domain": {
                        "name": "Permit2",
                        "chainId": 1,
                        "verifyingContract": "0x000000000022d473030f116ddee9f6b43ac78ba3",
                    },
                    "types": {
                        "PermitWitnessTransferFrom": [
                            {"name": "permitted", "type": "TokenPermissions"},
                            {"name": "spender", "type": "address"},
                            {"name": "nonce", "type": "uint256"},
                            {"name": "deadline", "type": "uint256"},
                            {"name": "witness", "type": "ExclusiveDutchOrder"},
                        ],
                        "TokenPermissions": [
                            {"name": "token", "type": "address"},
                            {"name": "amount", "type": "uint256"},
                        ],
                        "ExclusiveDutchOrder": [
                            {"name": "info", "type": "OrderInfo"},
                            {"name": "decayStartTime", "type": "uint256"},
                            {"name": "decayEndTime", "type": "uint256"},
                            {"name": "exclusiveFiller", "type": "address"},
                            {"name": "exclusivityOverrideBps", "type": "uint256"},
                            {"name": "inputToken", "type": "address"},
                            {"name": "inputStartAmount", "type": "uint256"},
                            {"name": "inputEndAmount", "type": "uint256"},
                            {"name": "outputs", "type": "DutchOutput[]"},
                        ],
                        "OrderInfo": [
                            {"name": "reactor", "type": "address"},
                            {"name": "swapper", "type": "address"},
                            {"name": "nonce", "type": "uint256"},
                            {"name": "deadline", "type": "uint256"},
                            {"name": "additionalValidationContract", "type": "address"},
                            {"name": "additionalValidationData", "type": "bytes"},
                        ],
                        "DutchOutput": [
                            {"name": "token", "type": "address"},
                            {"name": "startAmount", "type": "uint256"},
                            {"name": "endAmount", "type": "uint256"},
                            {"name": "recipient", "type": "address"},
                        ],
                    },
                    "values": {
                        "permitted": {
                            "token": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
                            "amount": {
                                "type": "BigNumber",
                                "hex": "0x056bc75e2d63100000",
                            },
                        },
                        "spender": "0x6000da47483062A0D734Ba3dc7576Ce6A0B645C4",
                        "nonce": {
                            "type": "BigNumber",
                            "hex": "0x04683295d60a099aaf9f1be560f73109eeaba6d826fe487999d1e8df70362b01",
                        },
                        "deadline": 1698670834,
                        "witness": {
                            "info": {
                                "reactor": "0x6000da47483062A0D734Ba3dc7576Ce6A0B645C4",
                                "swapper": "0x0000000000000000000000000000000000000000",
                                "nonce": {
                                    "type": "BigNumber",
                                    "hex": "0x04683295d60a099aaf9f1be560f73109eeaba6d826fe487999d1e8df70362b01",
                                },
                                "deadline": 1698670834,
                                "additionalValidationContract": "0x0000000000000000000000000000000000000000",
                                "additionalValidationData": "0x",
                            },
                            "decayStartTime": 1698670762,
                            "decayEndTime": 1698670822,
                            "exclusiveFiller": "0x32801aB1957Aaad1c65289B51603373802B4e8BB",
                            "exclusivityOverrideBps": {
                                "type": "BigNumber",
                                "hex": "0x64",
                            },
                            "inputToken": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
                            "inputStartAmount": {
                                "type": "BigNumber",
                                "hex": "0x056bc75e2d63100000",
                            },
                            "inputEndAmount": {
                                "type": "BigNumber",
                                "hex": "0x056bc75e2d63100000",
                            },
                            "outputs": [
                                {
                                    "token": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                                    "startAmount": {
                                        "type": "BigNumber",
                                        "hex": "0xb43d47962eb366",
                                    },
                                    "endAmount": {
                                        "type": "BigNumber",
                                        "hex": "0x8fe920f5eefe1f",
                                    },
                                    "recipient": "0x0000000000000000000000000000000000000000",
                                },
                                {
                                    "token": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                                    "startAmount": {
                                        "type": "BigNumber",
                                        "hex": "0x4550de620252",
                                    },
                                    "endAmount": {
                                        "type": "BigNumber",
                                        "hex": "0x37583eed62f6",
                                    },
                                    "recipient": "0x37a8f295612602f2774d331e562be9e61B83a327",
                                },
                            ],
                        },
                    },
                },
                "portionBips": 15,
                "portionAmount": "76213630665298",
                "portionRecipient": "0x37a8f295612602f2774d331e562be9e61B83a327",
            },
        },
        {
            "routing": "CLASSIC",
            "quote": {
                "methodParameters": {
                    "calldata": "0x3593564c000000000000000000000000000000000000000000000000000000000000006000000000000000000000000000000000000000000000000000000000000000a000000000000000000000000000000000000000000000000000000000653fab00000000000000000000000000000000000000000000000000000000000000000300060400000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000003000000000000000000000000000000000000000000000000000000000000006000000000000000000000000000000000000000000000000000000000000001800000000000000000000000000000000000000000000000000000000000000200000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000020000000000000000000000000000000000000000000000056bc75e2d6310000000000000000000000000000000000000000000000000000000c304e175dada6f00000000000000000000000000000000000000000000000000000000000000a00000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000002b6b175474e89094c44da98b954eedeac495271d0f0001f4c02aaa39b223fe8d0a0e5c4f27ead9083c756cc20000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000060000000000000000000000000c02aaa39b223fe8d0a0e5c4f27ead9083c756cc200000000000000000000000037a8f295612602f2774d331e562be9e61b83a327000000000000000000000000000000000000000000000000000000000000000f0000000000000000000000000000000000000000000000000000000000000060000000000000000000000000c02aaa39b223fe8d0a0e5c4f27ead9083c756cc2000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000c304e175dada6f",
                    "value": "0x00",
                    "to": "0x3fC91A3afd70395Cd496C647d5a6CC9D4B2b7FAD",
                },
                "blockNumber": "18463129",
                "amount": "100000000000000000000",
                "amountDecimals": "100",
                "quote": "55250327275170458",
                "quoteDecimals": "0.055250327275170458",
                "quoteGasAdjusted": "49183320797690822",
                "quoteGasAdjustedDecimals": "0.049183320797690822",
                "quoteGasAndPortionAdjusted": "49100445306778066",
                "quoteGasAndPortionAdjustedDecimals": "0.049100445306778066",
                "gasUseEstimateQuote": "6067006477479636",
                "gasUseEstimateQuoteDecimals": "0.006067006477479636",
                "gasUseEstimate": "198354",
                "gasUseEstimateUSD": "10.975434288112398323",
                "simulationStatus": "SUCCESS",
                "simulationError": False,
                "gasPriceWei": "30586761434",
                "route": [
                    [
                        {
                            "type": "v3-pool",
                            "address": "0x60594a405d53811d3BC4766596EFD80fd545A270",
                            "tokenIn": {
                                "chainId": 1,
                                "decimals": "18",
                                "address": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
                                "symbol": "DAI",
                            },
                            "tokenOut": {
                                "chainId": 1,
                                "decimals": "18",
                                "address": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                                "symbol": "WETH",
                            },
                            "fee": "500",
                            "liquidity": "1482804439370666457114890",
                            "sqrtRatioX96": "1863473130237137133038240165",
                            "tickCurrent": "-75002",
                            "amountIn": "100000000000000000000",
                            "amountOut": "55167451784257702",
                        }
                    ]
                ],
                "routeString": "[V3] 100.00% = DAI -- 0.05% [0x60594a405d53811d3BC4766596EFD80fd545A270] --> WETH",
                "quoteId": "5a22431c-12a5-4709-ae41-8a7a5d4e7381",
                "portionBips": 15,
                "portionRecipient": "0x37a8f295612602f2774d331e562be9e61B83a327",
                "portionAmount": "82875490912755",
                "portionAmountDecimals": "0.000082875490912755",
                "requestId": "2c5bc13a-e1af-458c-98ea-9b639630a7ea",
                "tradeType": "EXACT_INPUT",
                "slippage": 0.5,
            },
        },
    ],
}
