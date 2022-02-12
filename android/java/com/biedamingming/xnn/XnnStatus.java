package com.biedamingming.xnn;

/**
 * Created by Ethan on 2022-02-08.
 */

public class XnnStatus {
    public enum Status {
        XNN_SUCCESS(0),
        XNN_PARAM_ERROR(-1),
        XNN_INVALID_HANDLE(-2),
        XNN_INVALID_PIXEL_FORMAT(-3),
        XNN_FILE_NOT_FOUND(-4),
        XNN_INVALID_MODEL_FILE_FORMAT(-5),
        XNN_JNI_ERROR(-1001);

        private int status_;
        Status(int status) {
            status_ = status;
        }
        public int getStatus() {
            return status_;
        }
    }

    private int xnn_status_;
    XnnStatus(int status) {
        xnn_status_ = status;
    }
    public boolean equal(Status status) {
        return xnn_status_ == status.getStatus();
    }
    public Status getStatus() {
        for (Status s: Status.values()) {
            if (s.status_ == xnn_status_) {
                return s;
            }
        }
        return null;
    }
}
