tracer = {
    retVal: [],
    addrStack: [],
    byte2Hex: function (byte) {
        if (byte < 0x10) return '0' + byte.toString(16);
        return byte.toString(16);
    },
    array2Hex: function (arr) {
        var retVal = '';
        for (var i = 0; i < arr.length; i++) retVal += this.byte2Hex(arr[i]);
        return retVal;
    },
    getAddr: function (addr) {
        return '0x' + this.array2Hex(addr);
    },
    step: function (log, db) {
        if (this.addrStack.length === 0) {
            this.addrStack.push(this.getAddr(log.contract.getAddress()));
        }
        this.retVal.push({
            'pc': log.getPC(),
            'op': log.op.toString(),
            'depth': log.getDepth(),
            'address': this.addrStack[this.addrStack.length - 1],
            'is_error': false,
        });
    },
    fault: function (log, db) {
        this.retVal[this.retVal.length - 1]['is_error'] = true;
    },
    enter: function (cf) {
        this.addrStack.push(this.getAddr(cf.getTo()));
    },
    exit: function (fr) {
        this.addrStack.pop();
    },
    result: function (ctx, db) {
        return this.retVal;
    }
}