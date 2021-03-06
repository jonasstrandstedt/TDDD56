function y = time_difference_thread(col, line)
	NSEC_IN_SEC = 1000000000;
	MSEC_IN_SEC = 1000;
	NSEC_IN_MSEC = NSEC_IN_SEC / MSEC_IN_SEC;
	y = (((line(13) - line(11)) * NSEC_IN_SEC + line(14) - line(12))) / NSEC_IN_MSEC;
end
